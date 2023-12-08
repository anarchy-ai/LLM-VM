# AnarchyAI setup script
Write-Host -ForegroundColor Green "AnarchyAI setup script"

# Step 1: Check winget version
Write-Host "Checking winget version..."
winget --version
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "Error: Please update your Windows and then re-run the setup."
    exit
}

# Wait for a moment before proceeding
Start-Sleep -Seconds 1

# Step 2: Check Python version and install if not found
Write-Host "Checking Python version..."
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found. Installing Python 3.11..."
    winget install Python.Python.3.11
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Red "Error: Installing Python failed. Setup aborted."
        exit
    }
    
    # Wait for a moment before proceeding
    Start-Sleep -Seconds 3
    
    # Open a new PowerShell window in the same path
    Start-Process powershell -ArgumentList "-NoExit -Command Set-Location '$PWD'; Start-Sleep -Seconds 3; .\windows_setup.ps1"

    # Close the current PowerShell window
    exit
}

# Step 3: Create a virtual environment
Write-Host "Creating a virtual environment..."
python -m venv anarchyai
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "Error: Creating the virtual environment failed. Setup aborted."
    exit
}

# Wait for a moment before proceeding
Start-Sleep -Seconds 2

# Step 4: Activate the virtual environment
Write-Host "Activating the virtual environment..."
& .\anarchyai\Scripts\Activate
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "Error: Activating the virtual environment failed. Setup aborted."
    exit
}

# Wait for a moment before proceeding
Start-Sleep -Seconds 2

# Step 5: Install or upgrade pip
Write-Host "Installing or upgrading pip..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "Error: Installing or upgrading pip failed. Setup aborted."
    exit
}

# Step 6: Install dependencies in development mode
Write-Host "Installing dependencies in development mode..."
python -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "Error: Installing dependencies failed. Setup aborted."
    exit
}

Write-Host -ForegroundColor Green "Setup completed successfully."