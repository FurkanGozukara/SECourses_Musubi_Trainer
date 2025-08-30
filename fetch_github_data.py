import requests
import json
from datetime import datetime
import os

def fetch_github_data():
    owner = "kohya-ss"
    repo = "musubi-tuner"
    
    results = {
        "pulls": [],
        "issues": [],
        "commits": []
    }
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Fetch recent pull requests (both open and closed)
    print("Fetching pull requests...")
    for state in ["open", "closed"]:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {
            "state": state,
            "sort": "updated",
            "direction": "desc",
            "per_page": 30
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                pulls = response.json()
                for pr in pulls[:15]:  # Get 15 most recent
                    pr_data = {
                        "number": pr["number"],
                        "title": pr["title"],
                        "state": pr["state"],
                        "created_at": pr["created_at"],
                        "updated_at": pr["updated_at"],
                        "body": pr["body"],
                        "user": pr["user"]["login"],
                        "merged": pr.get("merged_at") is not None
                    }
                    
                    # Fetch PR comments
                    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr['number']}/comments"
                    comments_response = requests.get(comments_url, headers=headers)
                    if comments_response.status_code == 200:
                        pr_data["comments"] = [
                            {
                                "user": c["user"]["login"],
                                "body": c["body"],
                                "created_at": c["created_at"]
                            }
                            for c in comments_response.json()
                        ]
                    
                    # Fetch review comments
                    review_comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr['number']}/comments"
                    review_response = requests.get(review_comments_url, headers=headers)
                    if review_response.status_code == 200:
                        pr_data["review_comments"] = [
                            {
                                "user": rc["user"]["login"],
                                "body": rc["body"],
                                "path": rc.get("path", ""),
                                "line": rc.get("line", 0)
                            }
                            for rc in review_response.json()
                        ]
                    
                    results["pulls"].append(pr_data)
                    print(f"  Fetched PR #{pr['number']}: {pr['title']}")
        except Exception as e:
            print(f"Error fetching pull requests: {e}")
    
    # Fetch recent issues
    print("\nFetching issues...")
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {
        "state": "all",
        "sort": "updated",
        "direction": "desc",
        "per_page": 20
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            issues = response.json()
            for issue in issues[:10]:
                if "pull_request" not in issue:  # Skip PRs
                    issue_data = {
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "body": issue["body"],
                        "labels": [l["name"] for l in issue["labels"]],
                        "created_at": issue["created_at"],
                        "updated_at": issue["updated_at"]
                    }
                    results["issues"].append(issue_data)
                    print(f"  Fetched Issue #{issue['number']}: {issue['title']}")
    except Exception as e:
        print(f"Error fetching issues: {e}")
    
    # Fetch detailed commit information
    print("\nFetching recent commits...")
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {
        "per_page": 30
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            commits = response.json()
            for commit in commits[:20]:
                commit_data = {
                    "sha": commit["sha"],
                    "message": commit["commit"]["message"],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"],
                    "url": commit["html_url"]
                }
                results["commits"].append(commit_data)
                print(f"  Fetched commit: {commit['sha'][:7]} - {commit['commit']['message'][:50]}")
    except Exception as e:
        print(f"Error fetching commits: {e}")
    
    # Save results
    with open("github_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nData saved to github_analysis.json")
    print(f"Fetched: {len(results['pulls'])} PRs, {len(results['issues'])} issues, {len(results['commits'])} commits")
    
    # Analyze for Qwen-specific information
    print("\n=== Qwen-specific Analysis ===")
    qwen_mentions = []
    
    for pr in results["pulls"]:
        if any(keyword in (pr["title"] + " " + (pr["body"] or "")).lower() 
               for keyword in ["qwen", "qwen2", "vl", "vision", "image"]):
            qwen_mentions.append(f"PR #{pr['number']}: {pr['title']}")
            if pr.get("comments"):
                for comment in pr["comments"]:
                    if "qwen" in comment["body"].lower():
                        print(f"  Comment in PR #{pr['number']}: {comment['body'][:100]}...")
    
    for issue in results["issues"]:
        if any(keyword in (issue["title"] + " " + (issue["body"] or "")).lower()
               for keyword in ["qwen", "qwen2", "vl", "vision", "image"]):
            qwen_mentions.append(f"Issue #{issue['number']}: {issue['title']}")
    
    if qwen_mentions:
        print("\nQwen/Vision-related discussions found:")
        for mention in qwen_mentions:
            print(f"  - {mention}")
    
    return results

if __name__ == "__main__":
    fetch_github_data()