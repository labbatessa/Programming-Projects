# Import the necessary libraries/packages
from faunadb import query as q
from faunadb.client import FaunaClient
from csv import *
from datetime import datetime, date
import config as config

# Initialize the connection to Fauna
client = FaunaClient(secret=config.secret)

# Read the CSV file
with open("people-1000.csv") as csv:
    csvdata = reader(csv)

    # Skip the header row
    next(csvdata)

    # Obtain the age of each person in the file
    for row in csvdata:
        person_DOB = datetime.strptime(row[7], "%Y-%m-%d").date()
        age = (date.today().year - person_DOB.year)

        # Categorize the ages into age groups
        if age >= 0 and age <= 12:
            AgeGroup = "Child"
        if age >= 13 and age <= 17:
            AgeGroup = "Adolescent"
        if age >= 18 and age <= 59:
            AgeGroup = "Adult"
        if age >= 60:
            AgeGroup = "Senior"

        # Add the data to the FaunaDB database
        client.query(
            q.create(
                q.collection("Person"),
                {"data": {"Index": row[0], "User Id": row[1], "First Name": row[2], "Last Name": row[3], "Sex": row[4], "Email": row[5], "Phone": row[6], "Date of birth": row[7], "Job Title": row[8], "Age": age, "AgeGroup": AgeGroup}}
            )
        )
