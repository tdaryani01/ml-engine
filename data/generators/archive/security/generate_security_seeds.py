import csv
import urllib.request
import urllib.error
import urllib.parse
import os

GITHUB_SOURCES = {
    # --- 1. PROMPT INJECTION & JAILBREAKS (Target = 2: MALICIOUS) ---
    ("Direct_Injection", "Instruction_Override", 2): [
        "https://raw.githubusercontent.com/SwisskyRepo/PayloadsAllTheThings/master/Prompt Injection/README.md"
    ],
    ("Direct_Injection", "Roleplay_Jailbreak", 2): [
        "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/forbidden_question/forbidden_question_set.csv"
    ],
    ("Direct_Injection", "Encoded_Bypass", 2): [
        # Verified live endpoints for base64 payloads and bypass polyglots
        "https://raw.githubusercontent.com/minimaxir/big-list-of-naughty-strings/master/blns.base64.txt",
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/Polyglot/XSS-Polyglots.txt"
    ],

    # --- 2. TRADITIONAL WEB EXPLOITS & HEURISTICS (Target = 1: SUSPICIOUS) ---
    ("Traditional_Exploit", "SQL_Injection", 1): [
        "https://raw.githubusercontent.com/SwisskyRepo/PayloadsAllTheThings/master/SQL Injection/Intruder/Auth_Bypass.txt"
    ],
    ("Traditional_Exploit", "XSS_Payload", 1): [
        "https://raw.githubusercontent.com/SwisskyRepo/PayloadsAllTheThings/master/XSS Injection/Intruders/IntrudersXSS.txt"
    ],
    ("Traditional_Exploit", "Command_Injection", 1): [
        "https://raw.githubusercontent.com/SwisskyRepo/PayloadsAllTheThings/master/Command Injection/Intruder/command-execution-unix.txt",
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/command-injection-commix.txt"
    ],
    ("Traditional_Exploit", "Path_Traversal_LFI", 1): [
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/LFI/LFI-Jhaddix.txt"
    ],
    ("Traditional_Exploit", "SSTI_Payloads", 1): [
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/template-engines-special-vars.txt",
        "https://raw.githubusercontent.com/SwisskyRepo/PayloadsAllTheThings/master/Server Side Template Injection/README.md"
    ],

    # --- 3. BENIGN CONTROL HARD NEGATIVES FROM REPOS (Target = 0: CLEAN) ---
    ("Benign_Control", "JSON_Structure", 0): [
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/JSON.Fuzzing.txt"
    ],
    ("Benign_Control", "Format_Strings", 0): [
        # Verified live endpoints for format strings, naughty strings, and unicode
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/big-list-of-naughty-strings.txt",
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/Unicode.txt"
    ]
}

def fetch_payloads_from_urls(urls: list) -> list:
    headers = {'User-Agent': 'Mozilla/5.0'}
    collected_lines = []
    
    for raw_url in urls:
        encoded_url = urllib.parse.quote(raw_url, safe=':/%')
        try:
            req = urllib.request.Request(encoded_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8', errors='ignore')
                lines = [
                    line.strip() 
                    for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('//')
                ]
                if lines:
                    print(f"  [SUCCESS] Ingested {len(lines)} payloads from: {encoded_url}")
                    collected_lines.extend(lines)
        except Exception as e:
            print(f"  [ERROR] {encoded_url}: {e}")
            
    return collected_lines

def fetch_and_map_payloads(output_path: str = ".\\data\\samples\\csv\\github_ingested_seeds.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    seed_counter = 1
    rows = []
    
    print("Ingesting raw security payloads directly from GitHub repositories...")
    
    for (category_type, subcategory, target), urls in GITHUB_SOURCES.items():
        print(f"\nProcessing Node: ({category_type} -> {subcategory} | Target: {target})")
        lines = fetch_payloads_from_urls(urls)

        for payload in lines[:150]:
            seed_id = f"seed_gh_{seed_counter:04d}"
            prompt_str = payload if target == 2 else f"Input payload: {payload}"
            
            rows.append({
                "seed_id": seed_id,
                "category_type": category_type,
                "subcategory": subcategory,
                "prompt_text": prompt_str,
                "target": target
            })
            seed_counter += 1

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["seed_id", "category_type", "subcategory", "prompt_text", "target"]
        )
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"\nSuccessfully generated {len(rows)} total seeds in '{output_path}'!")

if __name__ == "__main__":
    fetch_and_map_payloads()