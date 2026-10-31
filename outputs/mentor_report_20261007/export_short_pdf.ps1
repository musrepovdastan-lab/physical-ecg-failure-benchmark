$ErrorActionPreference = 'Stop'
$shortDocx = Join-Path $PSScriptRoot 'ECG_MENTOR_UPDATE_SHORT_2026_10_07.docx'
$shortPdf = Join-Path $PSScriptRoot 'ECG_MENTOR_UPDATE_SHORT_2026_10_07.pdf'
$shortWord = $null
$shortDocument = $null
try {
    $shortWord = New-Object -ComObject Word.Application
    $shortWord.Visible = $false
    $shortWord.DisplayAlerts = 0
    $shortDocument = $shortWord.Documents.Open($shortDocx, $false, $true)
    $shortDocument.Repaginate()
    $shortDocument.ExportAsFixedFormat($shortPdf,17)
    Write-Output ('Pages: ' + $shortDocument.ComputeStatistics(2))
} finally {
    if ($null -ne $shortDocument) { $shortDocument.Close(0) }
    if ($null -ne $shortWord) { $shortWord.Quit() }
}
