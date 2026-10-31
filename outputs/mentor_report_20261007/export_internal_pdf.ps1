$ErrorActionPreference = 'Stop'
$reportDocx = Join-Path $PSScriptRoot 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.docx'
$reportPdf = Join-Path $PSScriptRoot 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.pdf'
$reportWord = $null
$reportDocument = $null
try {
    $reportWord = New-Object -ComObject Word.Application
    $reportWord.Visible = $false
    $reportWord.DisplayAlerts = 0
    $reportDocument = $reportWord.Documents.Open($reportDocx, $false, $true)
    $reportDocument.Repaginate()
    $reportDocument.ExportAsFixedFormat($reportPdf, 17)
    Write-Output ('Pages: ' + $reportDocument.ComputeStatistics(2))
} finally {
    if ($null -ne $reportDocument) { $reportDocument.Close(0) }
    if ($null -ne $reportWord) { $reportWord.Quit() }
}
