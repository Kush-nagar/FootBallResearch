import json
import csv
from openai import OpenAI

api_key = "API-KEY here"

client = OpenAI(api_key=api_key)

def load_player_profile(profile_id, profile_file):
    with open(profile_file, 'r') as file:
        profiles = json.load(file)
        for profile in profiles:
            if profile['id'] == profile_id:
                return profile
    return None

def load_reasons_csv(csv_file):
    reasons = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            reasons.append(row)
    return reasons

def load_solutions_csv(csv_file):
    solutions = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            solutions.append(row)
    return solutions

def get_top_reasons(profile, reasons):
    # Prepare the prompt
    prompt = f"""
You are an AI model specializing in sports psychology and performance analysis. Your task is to determine the top 5 most likely reasons affecting a soccer player's penalty performance based on their unique profile and a provided list of potential reasons.

Player Profile:
{json.dumps(profile, indent=2)}

Reasons List:
{json.dumps(reasons, indent=2)}

Based on the player's profile and the reasons provided, analyze and return the top 5 reasons that are most likely to have influenced their penalty performance. Provide these reasons in a ranked list format, from most likely to least likely.
"""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing sports performance."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content safely
    content = response.choices[0].message.content
    return content

def get_solutions(profile, reasons, solutions):
    # Prepare the prompt
    prompt = f"""
You are an AI model specializing in sports psychology and performance analysis. Your task is to determine the best solutions for each reason affecting a soccer player's penalty performance based on their unique profile and a provided list of possible solutions.

Player Profile:
{json.dumps(profile, indent=2)}

Reasons List:
{json.dumps(reasons, indent=2)}

Solutions List:
{json.dumps(solutions, indent=2)}

For each reason in the list, analyze and find the most appropriate solution(s) based on the player's profile. Provide these solutions in a clear, actionable format and do not provide the summary for the solutions at the end.
"""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing sports performance."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content safely
    content = response.choices[0].message.content
    return content

def main(id):
    profile_id = id
    profile_file = "Ronaldo_Profile.json"
    reasons_file = "reasons.csv"
    solutions_file = "solutions.csv"

    # Load player profile, reasons, and solutions
    profile = load_player_profile(profile_id, profile_file)
    if not profile:
        print("Profile not found!")
        return

    reasons = load_reasons_csv(reasons_file)
    solutions = load_solutions_csv(solutions_file)

    # Get top reasons from GPT
    top_reasons = get_top_reasons(profile, reasons)

    # Remove markdown (e.g., '**') from the reasons output for clean formatting
    clean_reasons = top_reasons.replace("**", "").strip()

    # Display the reasons in a clean format
    print("\nTop 5 Reasons Affecting the Player's Penalty Performance:\n")
    lines = clean_reasons.splitlines()
    reasons_list = []  # Collect reasons for passing to the solutions function
    for line in lines:
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
            reasons_list.append(line.strip())
            print(line.strip())
        else:
            print(f"{line.strip()}")

    # Get solutions for the reasons
    tailored_solutions = get_solutions(profile, reasons_list, solutions)

    # Remove markdown from solutions output and format them
    clean_solutions = (
        tailored_solutions.replace("**", "")
                          .replace("#", "")
                          .replace("\n\n", "\n")  # Remove excess newlines
                          .strip()
    )

    # Format the solutions like the reasons
    print("\nTailored Solutions for Each Reason:\n")
    lines = clean_solutions.splitlines()
    for line in lines:
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
            print(line.strip())  # Print the reason header
        else:
            print(f"{line.strip()}")  # Indent additional details
            


if __name__ == "__main__":
    main()