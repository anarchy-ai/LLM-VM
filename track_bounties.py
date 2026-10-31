import os
import requests
import csv
import sys

# Optional: set GITHUB_TOKEN environment variable to avoid rate limits
TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

REPO = "anarchy-ai/LLM-VM"
API_URL = f"https://api.github.com/repos/{REPO}/issues"

def fetch_closed_bounty_issues():
    issues_data = []
    page = 1
    while True:
        params = {"state": "closed", "per_page": 100, "page": page}
        response = requests.get(API_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error fetching issues: {response.status_code} - {response.text}")
            sys.exit(1)
            
        data = response.json()
        if not data:
            break
            
        for item in data:
            # Skip pull requests returned in issues endpoint if desired, or check labels
            labels = [lbl["name"].lower() for lbl in item.get("labels", [])]
            if any(tag in labels for tag in ["swag", "bounty", "good first issue"]):
                assignee = item.get("assignee")
                assignee_name = assignee["login"] if assignee else "Unassigned"
                closed_by = item.get("closed_by")
                closer_name = closed_by["login"] if closed_by else "Unknown"
                
                issues_data.append({
                    "issue_number": item["number"],
                    "title": item["title"],
                    "url": item["html_url"],
                    "assignee": assignee_name,
                    "closed_by": closer_name,
                    "labels": ", ".join(labels),
                    "closed_at": item.get("closed_at")
                })
        page += 1
    return issues_data

def export_to_csv(data, filename="bounty_tracking_report.csv"):
    fields = ["issue_number", "title", "url", "assignee", "closed_by", "labels", "closed_at"]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"[+] Successfully exported {len(data)} bounty/swag tracked items to {filename}")

if __name__ == "__main__":
    print(f"[*] Scanning closed issues for {REPO}...")
    bounty_issues = fetch_closed_bounty_issues()
    export_to_csv(bounty_issues)
