# HuggingFace Token Setup Script for PowerShell
# This script sets your HuggingFace token as an environment variable

Write-Host "=== HuggingFace Token Setup ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To get a token:" -ForegroundColor Yellow
Write-Host "1. Go to: https://huggingface.co/settings/tokens" -ForegroundColor White
Write-Host "2. Click 'New token'" -ForegroundColor White
Write-Host "3. Name it (e.g., 'llama-access')" -ForegroundColor White
Write-Host "4. Select 'Read' permissions" -ForegroundColor White
Write-Host "5. Copy the token (starts with 'hf_')" -ForegroundColor White
Write-Host ""

$token = Read-Host "Enter your HuggingFace token"

if ($token -match "^hf_") {
    # Set for current session
    $env:HUGGINGFACE_HUB_TOKEN = $token
    Write-Host ""
    Write-Host "✅ Token set for current PowerShell session" -ForegroundColor Green
    
    # Ask if user wants to set permanently
    $permanent = Read-Host "Set token permanently (requires restart)? (Y/n)"
    if ($permanent -eq "" -or $permanent -eq "Y" -or $permanent -eq "y") {
        [System.Environment]::SetEnvironmentVariable('HUGGINGFACE_HUB_TOKEN', $token, 'User')
        Write-Host "✅ Token set permanently (will be available after restarting terminal)" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Request access to Llama models at:" -ForegroundColor White
    Write-Host "   - https://huggingface.co/meta-llama/Llama-3-70B-Instruct" -ForegroundColor Yellow
    Write-Host "   - https://huggingface.co/meta-llama/Llama-3-8B-Instruct" -ForegroundColor Yellow
    Write-Host "2. Click 'Agree and access repository' on each page" -ForegroundColor White
    Write-Host "3. Wait for access approval (usually instant)" -ForegroundColor White
    Write-Host ""
    Write-Host "Your token is now set and will be used by the code!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Invalid token format. Token should start with 'hf_'" -ForegroundColor Red
    Write-Host "Please get a new token from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
}




