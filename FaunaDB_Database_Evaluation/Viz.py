# Import the necessary libraries/packages
from faunadb import query as q
from faunadb.client import FaunaClient
import config as config
import matplotlib.pyplot as plt
from collections import Counter

# Initialize the connection to Fauna
client = FaunaClient(secret=config.secret, domain="db.fauna.com")

# Initialize empty lists to store age groups and job titles
age_groups = []
job_titles = []

# Start pagination loop
next_page = None
while True:
    # Query the "Person" collection with pagination
    result = client.query(
        q.paginate(q.documents(q.collection("Person")), size=100, after=next_page)
    )

    # Extract the data from the result and add to the lists
    for ref in result["data"]:
        document = client.query(q.get(ref))

        age_group = document["data"]["AgeGroup"]
        age_groups.append(age_group)

        job_title = document["data"]["Job Title"]
        job_titles.append(job_title)

    # Get the next page cursor, if available
    next_page = result.get("after")
    if not next_page:
        break  # Exit the loop if there are no more pages

# Count the occurrences of each AgeGroup and Job Title
age_group_counts = Counter(age_groups)
job_title_counts = Counter(job_titles)

# Extract AgeGroup labels and counts for plotting
age_labels, age_counts = zip(*age_group_counts.items())

# Plot the number of people per AgeGroup
plt.figure(figsize=(8, 4))  # Set the figure size for AgeGroup visualization
plt.bar(age_labels, age_counts)
plt.xlabel('Age Group')
plt.ylabel('Number of People')
plt.title("Number of People per Age Group")
plt.show()

# Extract top 10 job titles and counts for plotting
top_job_titles = job_title_counts.most_common(10)
job_labels, job_counts = zip(*top_job_titles)

# Plot the top 10 most popular job titles
plt.figure(figsize=(10, 7))  # Set the figure size for Top Job Titles visualization
plt.barh(job_labels, job_counts)
plt.xlabel('Number of People')
plt.ylabel('Job Title')
plt.title("Top 10 Most Popular Job Titles")
plt.yticks(rotation=45)
plt.gca().invert_yaxis()  # Invert y-axis to display top job titles at the top
plt.tight_layout()
plt.show()

