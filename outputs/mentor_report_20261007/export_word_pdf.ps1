$ErrorActionPreference = 'Stop'
$reportDirectory = Join-Path $PSScriptRoot '.'
$reportDocx = [System.IO.Path]::GetFullPath((Join-Path $reportDirectory 'ECG_RESEARCH_COMPLETE_REPORT_2026_10_07.docx'))
$reportPdf = [System.IO.Path]::GetFullPath((Join-Path $reportDirectory 'ECG_RESEARCH_COMPLETE_REPORT_2026_10_07.pdf'))
$reportWord = $null
$reportDocument = $null
try {
    $reportWord = New-Object -ComObject Word.Application
    $reportWord.Visible = $false
    $reportWord.DisplayAlerts = 0
    $reportDocument = $reportWord.Documents.Open($reportDocx, $false, $false)
    foreach ($reportToc in $reportDocument.TablesOfContents) { $reportToc.Update() }
    $reportDocument.Fields.Update() | Out-Null
    $reportDocument.Repaginate()
    foreach ($reportToc in $reportDocument.TablesOfContents) { $reportToc.UpdatePageNumbers() }
    $reportDocument.Save()
    $reportDocument.ExportAsFixedFormat($reportPdf, 17)
    Write-Output ('Word PDF export completed: ' + $reportPdf)
    Write-Output ('Pages: ' + $reportDocument.ComputeStatistics(2))
} finally {
    if ($null -ne $reportDocument) { $reportDocument.Close(0) }
    if ($null -ne $reportWord) { $reportWord.Quit() }
}
