# Paths
$psEnvUtilPath = "./utils/Set-PsEnv.psm1"
$configPath = "./config/config.env"

# Import the util function
Import-Module -Name $psEnvUtilPath

# Executes the function which will place config variables temporarily into environment variables
Set-PsEnv $configPath

# Set re-tag timestamp
$retagTimestamp = $Env:retagTimestamp 
$o = [int](Get-Date -Date $retagTimestamp -UFormat %s -Millisecond 0) # Convert timestamp Unix Epoch Seconds; arg o

# Set path of conda environment's python.exe
$pythonPath = $Env:pythonPath

# Set start and end of work hours for logging purposes
$workEndPM = $Env:workEndHr - 12
$workStartAM = $Env:workStartHr

do {
    # Start Tagger
    $exitCode = (Start-Process -FilePath $pythonPath -ArgumentList "tagger.py -o $o" -NoNewWindow -Wait -PassThru).ExitCode

    # Signify done
    Write-Host "`n==============================="
    Write-Host "tagger.py exited with exit code ($exitCode)."
    switch ($exitCode) {
        0 {
            Write-Host "Tagger exited without problems.";
            Break
        }
        1 {
            Write-Host "Exception occurred in tagger.py script.";
            Break
        }
        2 {
            Write-Host "Missing/invalid arguments.";
            Break
        }
        3 {
            Write-Host "Tagger interrupted. (KeyboardInterrupt)";
            Break
        }
        5 {
            Write-Host "Tagging can only be done from $workEndPM PM to $workStartAM AM.";
            Break
        }
        99 {
            Write-Host "No more documents to process. Sleeping until the next day $workEndPM PM.";
            Break
        }
        500 {
            Write-Host "Elasticsearch connection failed to be established.";
            Break
        }
        Default {
            Write-Host "Unhandled exit code?? Something went wrong boi."
            Exit
        }
    }
    Write-Host "===============================`n"

    # Check if daemon should exit if fatal error
    if (-not ($exitCode -in @(0, 5, 99))) { # only these 3 indicate normal flow as of now, unless list is extended
        Write-Host "Exiting..."
        Exit # should exit... will end up in an infinite loop if it does not fix itself
    }

    # Set next time to awaken
    # if: no more documents to process AND day has not ended, then sleep till next day's 6 PM
    # else: sleep until current day's 6 PM
    if (($exitCode -eq 99) -and ((Get-Date -format %H) -in $Env:workEndHr..23)) {
        $sixPM = [DateTime]::Today.AddHours(24+$Env:workEndHr)
    } else {
        $sixPM = [DateTime]::Today.AddHours($Env:workEndHr)
    }

    Write-Host "Current time is $(Get-Date)"
    $waitTime = [math]::Round((New-TimeSpan -Start (Get-Date) -End $sixPM).TotalMinutes)

    # If there is time to wait, notify; 
    # Else it means it is already 6PM, continue to next iteration which should begin tagger.py operations
    if ($waitTime -gt 0) {
        Write-Host "Sleeping for $waitTime minutes till it is approximately after $workEndPM PM: $sixPM`n"
        Start-Sleep -Seconds ($waitTime * 60)
    }
    Write-Host "Restarting tagger...`n"
} while (1)