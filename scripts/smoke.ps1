<#
.SYNOPSIS
    阶段 4-6 · 一键契约冒烟（不依赖浏览器，验证后端契约与主链路）

.DESCRIPTION
    覆盖 docs/04-06 附录「一键冒烟脚本」的全部检查点：
    health → library → register（含重复 1004）→ recognize/image（含无脸 1002）
    → recognize/frame ×3 → library → delete。
    依赖：后端已在 -Api 指向的地址启动；样本图存在。

.PARAMETER Api
    后端 API 根地址，默认走直连 8000（与 frontend/.env.development 一致）。

.PARAMETER Name
    冒烟用的姓名，脚本执行前会先清理同名记录，保证首次采集必然成功。

.PARAMETER Keep
    保留冒烟采集的记录（默认脚本末尾删除，使库恢复原状）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1
#>
param(
    [string]$Api = 'http://127.0.0.1:8000/api',
    [string]$Name = '联调冒烟',
    [string]$Sample = '',
    [string]$NoFace = '',
    [switch]$Keep
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Sample) { $Sample = Join-Path $root 'backend/sample_faces/dingzhen/dingzhen_0.jpg' }
if (-not $NoFace) { $NoFace = Join-Path $root 'backend/sample_faces/_test_samples/no_face.jpg' }
$Api = $Api.TrimEnd('/')

Add-Type -AssemblyName System.Net.Http

$script:passed = 0
$script:failures = New-Object System.Collections.ArrayList

function Check([string]$label, [bool]$ok, [string]$detail = '') {
    if ($ok) {
        $script:passed++
        Write-Host ("  PASS  {0}" -f $label)
    }
    else {
        $suffix = if ($detail) { " :: $detail" } else { '' }
        [void]$script:failures.Add($label + $suffix)
        Write-Host ("  FAIL  {0}{1}" -f $label, $suffix)
    }
}

$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(60)
$utf8 = [System.Text.Encoding]::UTF8

function Invoke-Json([string]$method, [string]$path, $content = $null) {
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($method), ($Api + $path))
    if ($null -ne $content) { $req.Content = $content }
    $resp = $client.SendAsync($req).GetAwaiter().GetResult()
    $bytes = $resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
    $text = $utf8.GetString($bytes)
    $json = $null
    try { $json = $text | ConvertFrom-Json } catch { $json = $null }
    return [pscustomobject]@{ Status = [int]$resp.StatusCode; Text = $text; Json = $json }
}

function New-Multipart([hashtable]$fields, [string]$fieldName, [string]$filePath) {
    $mp = [System.Net.Http.MultipartFormDataContent]::new()
    foreach ($key in $fields.Keys) {
        $sc = [System.Net.Http.StringContent]::new([string]$fields[$key], $utf8)
        [void]$mp.Add($sc, $key)
    }
    if ($filePath) {
        $bytes = [System.IO.File]::ReadAllBytes($filePath)
        $bc = [System.Net.Http.ByteArrayContent]::new($bytes)
        $mediaType = if ($filePath -match '\.png$') { 'image/png' } else { 'image/jpeg' }
        # 赋值会回显到管道，必须吃掉，否则本函数返回 [MediaTypeHeaderValue, MultipartFormDataContent]
        $null = $bc.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new($mediaType)
        [void]$mp.Add($bc, $fieldName, [System.IO.Path]::GetFileName($filePath))
    }
    # MultipartFormDataContent 实现了 IEnumerable<HttpContent>，
    # 直接 `return $mp` 会被 PowerShell 展开成「各个 part」，必须用 -NoEnumerate 原样输出
    Write-Output -NoEnumerate $mp
}

Write-Host "== 阶段 4-6 契约冒烟 =="
Write-Host ("  API   : {0}" -f $Api)
Write-Host ("  Name  : {0}" -f $Name)
Write-Host ("  Sample: {0}" -f $Sample)
Write-Host ''

# 1) 健康检查
$health = Invoke-Json 'GET' '/health'
Check 'health HTTP 200' ($health.Status -eq 200) "status=$($health.Status)"
Check 'health code=0' ($health.Json.code -eq 0) $health.Text
Check 'health data.status=ok' ($health.Json.data.status -eq 'ok') $health.Text
Check 'health feature_dim=512' ($health.Json.data.feature_dim -eq 512) $health.Text
Write-Host ("  info  : device={0} library_size={1}" -f $health.Json.data.device, $health.Json.data.library_size)

# 2) 特征库列表
$before = Invoke-Json 'GET' '/face/library'
Check 'library code=0' ($before.Json.code -eq 0) $before.Text
Check 'library items 为数组' ($null -ne $before.Json.data.items) $before.Text
Check 'library total 与 items 数量一致' ([int]$before.Json.data.total -eq @($before.Json.data.items).Count) $before.Text
$preTotal = [int]$before.Json.data.total

# 3) 预清理同名记录，保证「首次采集」确定性
$pre = Invoke-Json 'DELETE' ('/face/library?name=' + [uri]::EscapeDataString($Name))
# 1005 = 目标不存在，属于「本来就是干净的」，不是失败
Check '预清理 delete 可用（1005 视为空库）' ($pre.Json.code -eq 0 -or $pre.Json.code -eq 1005) $pre.Text
if ([int]$pre.Json.data.deleted -gt 0) {
    Write-Host ("  info  : 已清理同名旧记录 {0} 条" -f $pre.Json.data.deleted)
}

# 4) 采集（multipart：file / name）
$reg = Invoke-Json 'POST' '/face/register/image' (New-Multipart @{ name = $Name } 'file' $Sample)
Check 'register code=0' ($reg.Json.code -eq 0) $reg.Text
if ($reg.Json.code -eq 0) {
    Check 'register 返回 record_id' ([int]$reg.Json.data.record_id -gt 0) $reg.Text
}

# 5) 重复采集必须被拒绝（1004）
$dup = Invoke-Json 'POST' '/face/register/image' (New-Multipart @{ name = $Name } 'file' $Sample)
Check 'register 重复 code=1004' ($dup.Json.code -eq 1004) $dup.Text

# 6) 图片识别（multipart：file / threshold / save_result）
$rec = Invoke-Json 'POST' '/recognize/image' (New-Multipart @{ threshold = '0.6'; save_result = 'false' } 'file' $Sample)
Check 'recognize/image code=0' ($rec.Json.code -eq 0) $rec.Text
# 库里若已有「同一张脸」的其它记录（例如数据脚本的张三用的就是本样本），
# 相似度同为 1.0，识别只返回 top-1，可能返回那条记录的名字 —— 不能用「名字必须等于 $Name」硬卡。
$hit = @($rec.Json.data.faces | Where-Object { $_.matched })
$self = @($hit | Where-Object { $_.name -eq $Name })
$alias = @($hit | Where-Object { $_.similarity -ge 0.99 })
Check 'recognize/image 命中本人（或库内同脸记录）' ($self.Count -ge 1 -or $alias.Count -ge 1) $rec.Text
$top = if ($self.Count -ge 1) { $self[0] } elseif ($alias.Count -ge 1) { $alias[0] } else { $null }
if ($null -ne $top) {
    Check 'similarity 落在 0~1' ($top.similarity -ge 0 -and $top.similarity -le 1) "similarity=$($top.similarity)"
    Check 'box 为 x/y/w/h' ($null -ne $top.box.x -and $null -ne $top.box.y -and $null -ne $top.box.w -and $null -ne $top.box.h) $rec.Text
}


# 7) 无脸图 → 1002
$noFaceResp = Invoke-Json 'POST' '/recognize/image' (New-Multipart @{ threshold = '0.6'; save_result = 'false' } 'file' $NoFace)
Check 'recognize/image(no face) code=1002' ($noFaceResp.Json.code -eq 1002) $noFaceResp.Text

# 8) 帧识别 ×3（串行，观察是否出现 429/1008）
$b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($Sample))
$frameBody = @{ image_base64 = "data:image/jpeg;base64,$b64"; threshold = 0.6 } | ConvertTo-Json -Compress
for ($i = 1; $i -le 3; $i++) {
    $jsonContent = [System.Net.Http.StringContent]::new($frameBody, $utf8, 'application/json')
    $frame = Invoke-Json 'POST' '/recognize/frame' $jsonContent
    Check ("recognize/frame #{0} HTTP 200" -f $i) ($frame.Status -eq 200) "status=$($frame.Status)"
    Check ("recognize/frame #{0} code=0" -f $i) ($frame.Json.code -eq 0) $frame.Text
}

# 9) 采集后特征库
$after = Invoke-Json 'GET' '/face/library'
$mine = @($after.Json.data.items | Where-Object { $_.name -eq $Name })
Check 'library 出现新记录' ($mine.Count -eq 1) ("count=" + $mine.Count)
Check 'library 每条 feature_dim=512' (@($after.Json.data.items | Where-Object { $_.feature_dim -ne 512 }).Count -eq 0) $after.Text
Check 'library total 增加 1' ([int]$after.Json.data.total -eq ($preTotal + 1)) ("total=" + $after.Json.data.total)

# 10) 删除（query 参数 name）
$del = Invoke-Json 'DELETE' ('/face/library?name=' + [uri]::EscapeDataString($Name))
Check 'delete code=0' ($del.Json.code -eq 0) $del.Text
Check 'delete deleted>=1' ([int]$del.Json.data.deleted -ge 1) $del.Text

$final = Invoke-Json 'GET' '/face/library'
Check 'delete 后列表无该姓名' (@($final.Json.data.items | Where-Object { $_.name -eq $Name }).Count -eq 0) $final.Text

if ($Keep) {
    $again = Invoke-Json 'POST' '/face/register/image' (New-Multipart @{ name = $Name } 'file' $Sample)
    Write-Host ("  info  : -Keep 生效，已重新采集 {0}（code={1}）" -f $Name, $again.Json.code)
}

Write-Host ''
Write-Host ("SUMMARY: {0} passed, {1} failed" -f $script:passed, $script:failures.Count)
if ($script:failures.Count -gt 0) {
    $script:failures | ForEach-Object { Write-Host ("  - {0}" -f $_) }
    exit 1
}
exit 0

