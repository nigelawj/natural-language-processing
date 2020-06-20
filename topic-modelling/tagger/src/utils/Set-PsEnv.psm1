<#
Credits: rajivharris: https://github.com/rajivharris/Set-PsEnv
Exports environment variable from the .env file to the current process.

This function has slight modifications from the original.

NOTES:
To assign a value, use the "=" operator; e.g. <variable name> = <value>

To prefix a value to an existing env variable, use the ":=" operator; e.g. <variable name> := <value>
To suffix a value to an existing env variable, use the "=:" operator; e.g. <variable name> =: <value>
#>

function Set-PsEnv {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
    param(
        [Parameter()]
        [string]$localEnvFile
    )

    # return if no env file
    if (!(Test-Path $localEnvFile)) {
        Write-Verbose "No .env file"
        return
    }

    # read the local env file
    $content = Get-Content $localEnvFile -ErrorAction Stop
    Write-Verbose "Parsed .env file"

    # load the content to environment
    foreach ($line in $content) {
        if([string]::IsNullOrWhiteSpace($line)){
            Write-Verbose "Skipping empty line"
            continue
        }

        # ignore comments
        if($line.StartsWith("#")){
            Write-Verbose "Skipping comment: $line"
            continue
        }

        # get the operator
        if($line -like "*:=*"){
            Write-Verbose "Prefix"
            $kvp = $line -split ":=",2
            $cmd = '$Env:{0} = "{1};$Env:{0}"' -f $kvp[0].Trim(),$kvp[1].Trim()
        }
        elseif ($line -like "*=:*"){
            Write-Verbose "Suffix"
            $kvp = $line -split "=:",2
            $cmd = '$Env:{0} += ";{1}"' -f $kvp[0].Trim(),$kvp[1].Trim()
        }
        else {
            Write-Verbose "Assign"
            $kvp = $line -split "=",2
            $cmd = '$Env:{0} = "{1}"' -f $kvp[0].Trim(),$kvp[1].Trim()
        }

        Write-Verbose $cmd
        
        if ($PSCmdlet.ShouldProcess("$($cmd)", "Execute")) {
            Invoke-Expression $cmd
        }
    }
}

Export-ModuleMember -Function Set-PsEnv