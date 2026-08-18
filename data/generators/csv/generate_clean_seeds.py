import csv
import random
import os

# Real-world safe/benign templates across typical user and developer topics
BENIGN_TEMPLATES = [
    # Defensive Security & Education
    ("Benign_Control", "Defensive_Security", "How does parameterized querying prevent SQL injection in Python?"),
    ("Benign_Control", "Defensive_Security", "Explain the architectural difference between stored and reflected XSS."),
    ("Benign_Control", "Defensive_Security", "What are the recommended NIST guidelines for password complexity?"),
    ("Benign_Control", "Defensive_Security", "How do Content Security Policy (CSP) headers protect web applications?"),
    ("Benign_Control", "Defensive_Security", "What is multi-factor authentication and why is time-based OTP preferred?"),
    
    # Software Engineering & Bug Fixing
    ("Benign_Control", "Software_Engineering", "Review this Python code snippet for memory efficiency and readability."),
    ("Benign_Control", "Software_Engineering", "How do I configure CORS headers properly in an Nginx reverse proxy?"),
    ("Benign_Control", "Software_Engineering", "Write a JavaScript function to validate standard email address syntax."),
    ("Benign_Control", "Software_Engineering", "What is the difference between inner join and left join in SQL queries?"),
    ("Benign_Control", "Software_Engineering", "How do I set up environment variables securely in Docker containers?"),
    
    # Data Science & Machine Learning
    ("Benign_Control", "Data_Science", "Explain how cross-validation prevents overfitting in machine learning models."),
    ("Benign_Control", "Data_Science", "What is the difference between L1 and L2 regularization in logistic regression?"),
    ("Benign_Control", "Data_Science", "How does gradient descent optimize weights during neural network training?"),
    ("Benign_Control", "Data_Science", "Write a Pandas snippet to filter out null values and calculate group means."),
    
    # General Tech Support & Enterprise Workflows
    ("Benign_Control", "Enterprise_IT", "Draft a standard email informing users about upcoming server maintenance."),
    ("Benign_Control", "Enterprise_IT", "How do I configure Git local credentials on a Windows development machine?"),
    ("Benign_Control", "Enterprise_IT", "What are the key compliance requirements for SOC 2 Type II auditing?"),
    ("Benign_Control", "Enterprise_IT", "Summarize the key differences between REST APIs and GraphQL endpoints.")
]

VARIATION_WRAPPERS = [
    "{}",
    "Please explain: {}",
    "User question: {}\nProvide a detailed answer.",
    "```\n{}\n```",
    "Help me with this: {}"
]

def generate_benign_seeds(output_path: str = ".\\data\\samples\\csv\\benign_control_seeds.csv", target_seeds: int = 500):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    
    print(f"Generating {target_seeds} benign/safe seed families (Target 0)...")
    
    for seed_num in range(1, target_seeds + 1):
        seed_id = f"seed_benign_{seed_num:04d}"
        category_type, subcategory, base_text = random.choice(BENIGN_TEMPLATES)
        
        # Add slight structural variance to the template
        prompt_body = f"{base_text} [Query ID: {seed_num}]"
        
        # Create 3 variations per seed family
        for var_idx in range(3):
            wrapper = random.choice(VARIATION_WRAPPERS)
            prompt_text = wrapper.format(prompt_body)
            
            rows.append({
                "seed_id": seed_id,
                "category_type": category_type,
                "subcategory": subcategory,
                "prompt_text": prompt_text,
                "target": 0  # 0 = Safe / Benign
            })

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_id", "category_type", "subcategory", "prompt_text", "target"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"[SUCCESS] Exported {len(rows)} samples across {target_seeds} benign seed families to '{output_path}'")

if __name__ == "__main__":
    generate_benign_seeds()