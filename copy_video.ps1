$files = Get-ChildItem 'c:\Users\sarke\Desktop\SIh prototype\' | Where-Object { $_.Extension -eq '.mp4' }
foreach ($f in $files) {
    Write-Output "Found: $($f.FullName)"
    Copy-Item $f.FullName 'c:\Users\sarke\Desktop\SIh prototype\public\ocean_bg.mp4' -Force
    Write-Output "Copied OK"
}
