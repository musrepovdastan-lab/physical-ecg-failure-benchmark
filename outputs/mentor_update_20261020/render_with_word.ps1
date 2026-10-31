$ErrorActionPreference = 'Stop'
$reportPath = Join-Path $PSScriptRoot 'ECG_Research_Update_After_Mentor_Feedback_2026_10_20.docx'
$qaPath = Join-Path $PSScriptRoot 'qa'
$pdfPath = Join-Path $qaPath 'ECG_Research_Update_After_Mentor_Feedback_2026_10_20.pdf'
New-Item -ItemType Directory -Path $qaPath -Force | Out-Null
$reportWord = $null
$reportDocument = $null
try {
    $reportWord = New-Object -ComObject Word.Application
    $reportWord.Visible = $false
    $reportWord.DisplayAlerts = 0
    $reportDocument = $reportWord.Documents.Open($reportPath, $false, $true)
    $reportDocument.Repaginate()
    $reportDocument.ExportAsFixedFormat($pdfPath, 17)
    Write-Output ('Pages: ' + $reportDocument.ComputeStatistics(2))
} finally {
    if ($null -ne $reportDocument) { $reportDocument.Close(0) }
    if ($null -ne $reportWord) { $reportWord.Quit() }
}
$popplerPath = 'C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
& $popplerPath -r 120 -png $pdfPath (Join-Path $qaPath 'page')
if ($LASTEXITCODE -ne 0) { throw 'PDF page rendering failed' }
