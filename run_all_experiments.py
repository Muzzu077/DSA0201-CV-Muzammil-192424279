import os
import subprocess
import sys
import glob

def main():
    print("==================================================")
    print("Running all Computer Vision Lab Experiments...")
    print("==================================================")
    
    # Find all experiment scripts
    scripts = sorted(glob.glob("Exp_*.py"))
    
    if not scripts:
        print("No experiment scripts found in the current directory!")
        return
        
    print(f"Found {len(scripts)} scripts to execute.\\n")
    
    results = []
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    for script in scripts:
        print(f"Running {script}...", end="", flush=True)
        try:
            # Run the script. We use a timeout of 10 seconds per script to prevent hanging.
            # We set environment variables if needed, or just run it.
            res = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                print(" SUCCESS")
                results.append((script, "Success", ""))
            else:
                print(" FAILED")
                error_msg = res.stderr.strip() or res.stdout.strip()
                results.append((script, "Failed", error_msg))
        except subprocess.TimeoutExpired:
            print(" TIMEOUT (15s)")
            results.append((script, "Timeout", "Execution took longer than 15 seconds"))
        except Exception as e:
            print(" ERROR")
            results.append((script, "Error", str(e)))
            
    print("\\n==================================================")
    print("Execution Summary:")
    print("==================================================")
    
    success_count = sum(1 for r in results if r[1] == "Success")
    failed_count = len(results) - success_count
    
    for script, status, error in results:
        status_str = f"[{status}]"
        if status != "Success":
            print(f"{script:<40} {status_str:<10} - Error: {error}")
        else:
            print(f"{script:<40} {status_str:<10}")
            
    print("\\n--------------------------------------------------")
    print(f"Total: {len(results)} | Success: {success_count} | Failed: {failed_count}")
    print("--------------------------------------------------")
    
    # Write a summary log file
    with open("execution_report.txt", "w") as f:
        f.write("==================================================\\n")
        f.write("Computer Vision Lab Experiments Execution Report\\n")
        f.write("==================================================\\n\\n")
        f.write(f"Total Scripts: {len(results)}\\n")
        f.write(f"Success: {success_count}\\n")
        f.write(f"Failed: {failed_count}\\n\\n")
        f.write("Details:\\n")
        f.write("-" * 60 + "\\n")
        for script, status, error in results:
            if status == "Success":
                f.write(f"{script:<40} {status:<10}\\n")
            else:
                f.write(f"{script:<40} {status:<10} - Error: {error}\\n")
                
    print("Detailed report saved to execution_report.txt")

if __name__ == "__main__":
    main()
