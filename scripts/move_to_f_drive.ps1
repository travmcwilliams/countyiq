# Move CountyIQ project from C:\Projects\countyiq to F:\Projects\countyiq
# Run from C:\Projects\countyiq as Administrator if needed

$source = "C:\Projects\countyiq"
$dest = "F:\Projects\countyiq"

Write-Host "Moving CountyIQ project to F: drive..." -ForegroundColor Cyan
Write-Host "Source: $source" -ForegroundColor Yellow
Write-Host "Destination: $dest" -ForegroundColor Yellow

# Check if source exists
if (-not (Test-Path $source)) {
    Write-Host "ERROR: Source directory not found: $source" -ForegroundColor Red
    exit 1
}

# Create F:\Projects if it doesn't exist
if (-not (Test-Path "F:\Projects")) {
    Write-Host "Creating F:\Projects..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path "F:\Projects" -Force | Out-Null
}

# Check if destination already exists
if (Test-Path $dest) {
    Write-Host "WARNING: Destination already exists: $dest" -ForegroundColor Yellow
    $response = Read-Host "Overwrite? (y/N)"
    if ($response -ne "y") {
        Write-Host "Aborted." -ForegroundColor Red
        exit 1
    }
    Remove-Item -Path $dest -Recurse -Force
}

# Copy everything (preserving git, permissions, etc.)
Write-Host "Copying project files (this may take a few minutes)..." -ForegroundColor Cyan
robocopy $source $dest /E /COPYALL /R:3 /W:5 /NP /NFL /NDL

if ($LASTEXITCODE -ge 8) {
    Write-Host "ERROR: Copy failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# Verify git still works
Write-Host "Verifying git repository..." -ForegroundColor Cyan
Push-Location $dest
try {
    $gitStatus = git status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Git repository verified" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Git status check failed" -ForegroundColor Yellow
        Write-Host $gitStatus
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "✓ Migration complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Open F:\Projects\countyiq in Cursor (File → Open Folder)" -ForegroundColor White
Write-Host "2. Verify everything works: cd F:\Projects\countyiq; git status" -ForegroundColor White
Write-Host "3. Once verified, you can delete C:\Projects\countyiq to free space" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT: Update your workspace in Cursor to F:\Projects\countyiq" -ForegroundColor Yellow
