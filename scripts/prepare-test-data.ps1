<#
.SYNOPSIS
    阶段 4-6 · 联调测试数据准备（一键切换 空库 / 单人 / 多人）

.DESCRIPTION
    按 docs/04-06 §4.2 落地，三种状态可相互切换：
      empty  → 清空 face_features（E2E-02 空库识别、E2E-15 空态）
      single → 张三 1 条（采集后命中链路）
      multi  → 张三 2 条 + 李四 1 条 = 3 条 / 2 人（聚合视图、按人删除）

    说明：同名重复录入会被后端拒绝（1004），所以「一人多条」只能用 SQL 直插副本
    （复制 id 最小的那条，即 first 状态下录入的张三）。

.PARAMETER State
    empty | single | multi，默认 multi（E2E-12 聚合视图所需）

.PARAMETER DbPath
    特征库路径，默认 backend/face_features.db

.PARAMETER Python
    执行 SQL 的解释器，默认 PATH 中的 python（只用标准库 sqlite3，无需 cv_demo）

.PARAMETER NoBackup
    切换前不备份库文件（默认备份为 face_features.db.bak-yyyyMMdd-HHmmss）

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/prepare-test-data.ps1 -State empty

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/prepare-test-data.ps1 -State multi
#>
param(
    [ValidateSet('empty', 'single', 'multi')]
    [string]$State = 'multi',
    [string]$Api = 'http://127.0.0.1:8000/api',
    [string]$DbPath = '',
    [string]$Python = 'python',
    [string]$SampleA = '',
    [string]$SampleB = '',
    [string]$NameA = '张三',
    [string]$NameB = '李四',
    [switch]$NoBackup
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $DbPath) { $DbPath = Join-Path $root 'backend/face_features.db' }
if (-not $SampleA) { $SampleA = Join-Path $root 'backend/sample_faces/dingzhen/dingzhen_0.jpg' }
if (-not $SampleB) { $SampleB = Join-Path $root 'backend/sample_faces/coke/coke_0.jpg' }
if (-not (Test-Path $DbPath)) { throw "特征库文件不存在: $DbPath" }
$Api = $Api.TrimEnd('/')

Add-Type -AssemblyName System.Net.Http
$utf8 = [System.Text.Encoding]::UTF8
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(60)

function Invoke-Api([string]$method, [string]$path, $content = $null) {
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($method), ($Api + $path))
    if ($null -ne $content) { $req.Content = $content }
    $resp = $client.SendAsync($req).GetAwaiter().GetResult()
    $text = $utf8.GetString($resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult())
    $json = $null
    try { $json = $text | ConvertFrom-Json } catch { $json = $null }
    return [pscustomobject]@{ Status = [int]$resp.StatusCode; Text = $text; Json = $json }
}

function New-Multipart([string]$name, [string]$filePath) {
    $mp = [System.Net.Http.MultipartFormDataContent]::new()
    [void]$mp.Add([System.Net.Http.StringContent]::new($name, $utf8), 'name')
    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    $bc = [System.Net.Http.ByteArrayContent]::new($bytes)
    $null = $bc.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new('image/jpeg')
    [void]$mp.Add($bc, 'file', [System.IO.Path]::GetFileName($filePath))
    # MultipartFormDataContent 是可枚举集合，直接 return 会被展开成各个 part
    Write-Output -NoEnumerate $mp
}

function Invoke-Sql([string]$sql) {
    $code = 'import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); c.executescript(sys.argv[2]); c.commit(); c.close()'
    $out = & $Python -c $code $DbPath $sql 2>&1
    if ($LASTEXITCODE -ne 0) { throw ("SQL 执行失败: {0}`n{1}" -f $sql, ($out -join "`n")) }
}

Write-Host "== 阶段 4-6 测试数据准备 =="
Write-Host ("  State : {0}" -f $State)
Write-Host ("  DB    : {0}" -f $DbPath)
Write-Host ("  API   : {0}" -f $Api)

# 1) 备份（直接改库前留一份，便于回滚）
if (-not $NoBackup) {
    $backup = '{0}.bak-{1}' -f $DbPath, (Get-Date -Format 'yyyyMMdd-HHmmss')
    Copy-Item $DbPath $backup
    Write-Host ("  备份  : {0}" -f $backup)
}

# 2) 清空
Invoke-Sql 'DELETE FROM face_features;'
Write-Host '  已清空 face_features'

# 3) 按状态录入（走接口，与前端 multipart 完全同一路径）
$failed = 0
if ($State -ne 'empty') {
    $regA = Invoke-Api 'POST' '/face/register/image' (New-Multipart $NameA $SampleA)
    if ($regA.Json.code -eq 0) {
        Write-Host ("  录入  : {0}（record_id={1}）" -f $NameA, $regA.Json.data.record_id)
    }
    else {
        Write-Host ("  FAIL  : 录入 {0} → code={1} {2}" -f $NameA, $regA.Json.code, $regA.Json.message)
        $failed++
    }
}

if ($State -eq 'multi') {
    $regB = Invoke-Api 'POST' '/face/register/image' (New-Multipart $NameB $SampleB)
    if ($regB.Json.code -eq 0) {
        Write-Host ("  录入  : {0}（record_id={1}）" -f $NameB, $regB.Json.data.record_id)
    }
    else {
        Write-Host ("  FAIL  : 录入 {0} → code={1} {2}" -f $NameB, $regB.Json.code, $regB.Json.message)
        $failed++
    }
    # 复制 id 最小的一行（即张三）制造「一人多条」；纯 ASCII SQL，避免编码问题
    Invoke-Sql 'INSERT INTO face_features (feature, timestamp, face_id, feature_dim) SELECT feature, timestamp, face_id, feature_dim FROM face_features ORDER BY id ASC LIMIT 1;'
    Write-Host '  直插  : 已复制首条记录（同人多条）'
}

# 4) 校验
$lib = Invoke-Api 'GET' '/face/library'
$items = @($lib.Json.data.items)
$people = @($items | Select-Object -ExpandProperty name -Unique)
$total = [int]$lib.Json.data.total

Write-Host ''
Write-Host ("特征库现状: total={0}, people={1}" -f $total, $people.Count)
$items | ForEach-Object { Write-Host ("  - id={0} name={1} timestamp={2} dim={3}" -f $_.id, $_.name, $_.timestamp, $_.feature_dim) }

$expectTotal = switch ($State) { 'empty' { 0 } 'single' { 1 } 'multi' { 3 } }
$expectPeople = switch ($State) { 'empty' { 0 } 'single' { 1 } 'multi' { 2 } }
if ($total -ne $expectTotal) {
    Write-Host ("  FAIL  total 期望 {0}，实际 {1}" -f $expectTotal, $total)
    $failed++
}
if ($people.Count -ne $expectPeople) {
    Write-Host ("  FAIL  people 期望 {0}，实际 {1}" -f $expectPeople, $people.Count)
    $failed++
}
if (@($items | Where-Object { $_.feature_dim -ne 512 }).Count -gt 0) {
    Write-Host '  FAIL  存在 feature_dim != 512 的记录'
    $failed++
}
if ($State -eq 'multi') {
    $countA = @($items | Where-Object { $_.name -eq $NameA }).Count
    $countB = @($items | Where-Object { $_.name -eq $NameB }).Count
    if ($countA -ne 2 -or $countB -ne 1) {
        Write-Host ("  FAIL  {0} 期望 2 条、{1} 期望 1 条，实际 {2}/{3}" -f $NameA, $NameB, $countA, $countB)
        $failed++
    }
}

Write-Host ''
if ($failed -gt 0) {
    Write-Host ("RESULT: {0} 项校验失败" -f $failed)
    exit 1
}
Write-Host ("RESULT: OK —— 数据状态 [{0}] 就绪" -f $State)
exit 0

