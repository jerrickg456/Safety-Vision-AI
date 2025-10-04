# PowerShell script to package submission for hackathon

$ErrorActionPreference = "Stop"

# Create temporary directory for packaging
$tempDir = "temp_submission"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

Write-Host "Packaging submission files..."

# Copy required files and directories
$itemsToCopy = @(
    "scripts/train.py",
    "scripts/predict.py", 
    "app/",
    "data/config.yaml",
    "README.md"
)

foreach ($item in $itemsToCopy) {
    if (Test-Path $item) {
        $dest = Join-Path $tempDir $item
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        if (Test-Path $item -PathType Container) {
            Copy-Item -Recurse $item $dest
        } else {
            Copy-Item $item $dest
        }
        Write-Host "Copied: $item"
    } else {
        Write-Warning "File not found: $item"
    }
}

# Copy model weights if available
if (Test-Path "runs/baseline/weights/best.pt") {
    $weightsDir = Join-Path $tempDir "runs/baseline/weights"
    New-Item -ItemType Directory -Path $weightsDir -Force | Out-Null
    Copy-Item "runs/baseline/weights/best.pt" $weightsDir
    Write-Host "Copied: runs/baseline/weights/best.pt"
} else {
    Write-Warning "Model weights not found: runs/baseline/weights/best.pt"
}

# Copy training results and plots
if (Test-Path "runs/baseline") {
    $resultsItems = @("results.png", "confusion_matrix.png", "labels.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg")
    foreach ($resultItem in $resultsItems) {
        $resultPath = "runs/baseline/$resultItem"
        if (Test-Path $resultPath) {
            $destPath = Join-Path $tempDir "runs/baseline/$resultItem"
            $destDir = Split-Path $destPath -Parent
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item $resultPath $destPath
            Write-Host "Copied: $resultPath"
        }
    }
}

# Create zip file
$zipPath = "submission.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($tempDir, $zipPath)

# Cleanup
Remove-Item -Recurse -Force $tempDir

Write-Host "Submission package created: $zipPath"
Write-Host "Package contents:"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$zip.Entries | ForEach-Object { Write-Host "  $($_.FullName)" }
$zip.Dispose()