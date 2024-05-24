$trackerPath = "siamfc\siamfc_lt.py"
$resultsPath = "results_lt\"
$content = Get-Content $trackerPath -Encoding UTF8

$samples = (10, 20, 30, 40, 50, 60)
$precisions = [System.Collections.ArrayList]@()
$recalls = [System.Collections.ArrayList]@()
$f1s = [System.Collections.ArrayList]@()
foreach($s in $samples){
    if(Test-Path $resultsPath){
        Get-ChildItem -Path $resultsPath -Recurse | Remove-Item -Force -Recurse
        #Remove-Item $resultsPath -Force
    }
    Write-Host "s=$s"
    $newRow = "            'num_samples': $s,"
    $content[97] = $newRow
    $content | Out-File $trackerPath -Encoding utf8 -Force
    $content
    python run_tracker.py --dataset . --net siamfc_net.pth --results_dir results_lt  --visualize
    $expr = "python performance_evaluation.py --dataset . --results_dir results_st"
    $res = Invoke-Expression $expr
    $precision = [float]$res[1].split(": ", [System.StringSplitOptions]::RemoveEmptyEntries)[1].trim()
    $recall = [float]$res[2].split(": ", [System.StringSplitOptions]::RemoveEmptyEntries)[1].trim()
    $f1 = [float]$res[3].split(": ", [System.StringSplitOptions]::RemoveEmptyEntries)[1].trim()
    [void]$precisions.Add($precision)
    [void]$recalls.Add($recall)
    [void]$f1s.Add($f1)
}
Write-Host
$samples | Write-Host
$precisions | ForEach-Object {[math]::Round($_, 2)}
$recalls | ForEach-Object {[math]::Round($_, 2)}
$f1s | ForEach-Object {[math]::Round($_, 2)}