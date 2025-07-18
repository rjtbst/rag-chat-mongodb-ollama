# import os
# from pymongo import MongoClient
# from datetime import datetime, timedelta
# from faker import Faker
# from bson import ObjectId
# import random
# from dotenv import load_dotenv

# load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")

# # MongoDB setup
# client = MongoClient(MONGO_URI)
# db = client.sample_mflix
# fake = Faker()
# random.seed(42)

# # Drop old data
# db.users.drop()
# db.products.drop()
# db.categories.drop()
# db.orders.drop()
# db.payments.drop()
# db.reviews.drop()
# db.carts.drop()
# db.wishlists.drop()

# # --- Categories ---
# category_names = ["Electronics", "Clothing", "Books", "Home Appliances", "Sports", "Premium"]
# category_ids = {}
# for name in category_names:
#     cat = {
#         "_id": ObjectId(),
#         "name": name,
#         "description": f"{name} related items"
#     }
#     db.categories.insert_one(cat)
#     category_ids[name] = cat["_id"]

# # --- Products ---
# product_ids = []
# for _ in range(100):
#     cat_name = random.choice(category_names)
#     prod = {
#         "_id": ObjectId(),
#         "name": fake.word().capitalize() + " " + fake.word().capitalize(),
#         "description": fake.sentence(),
#         "categoryId": category_ids[cat_name],
#         "price": round(random.uniform(10, 2000), 2),
#         "stock": random.randint(0, 100),
#         "images": [fake.image_url() for _ in range(random.randint(1, 3))],
#         "createdAt": datetime.now() - timedelta(days=random.randint(0, 180)),
#         "updatedAt": datetime.now()
#     }
#     db.products.insert_one(prod)
#     product_ids.append(prod["_id"])

# # --- Users ---
# user_ids = []
# for _ in range(50):
#     user = {
#         "_id": ObjectId(),
#         "name": fake.name(),
#         "email": fake.email(),
#         "phone": fake.phone_number(),
#         "address": {
#             "country": "India",
#             "city": fake.city(),
#             "zip": fake.postcode()
#         },
#         "createdAt": datetime.now() - timedelta(days=random.randint(0, 365)),
#         "lastLogin": datetime.now() - timedelta(days=random.randint(0, 60))
#     }
#     db.users.insert_one(user)
#     user_ids.append(user["_id"])

# # --- Carts ---
# for user_id in user_ids:
#     cart = {
#         "_id": ObjectId(),
#         "userId": user_id,
#         "items": [{
#             "productId": random.choice(product_ids),
#             "quantity": random.randint(1, 5),
#             "priceAtTime": round(random.uniform(50, 1500), 2)
#         } for _ in range(random.randint(1, 3))],
#         "updatedAt": datetime.now()
#     }
#     db.carts.insert_one(cart)

# # --- Orders & Payments ---
# order_ids = []
# for _ in range(100):
#     user_id = random.choice(user_ids)
#     order_items = [{
#         "productId": random.choice(product_ids),
#         "quantity": random.randint(1, 3),
#         "unitPrice": round(random.uniform(50, 1500), 2)
#     } for _ in range(random.randint(1, 3))]

#     total = sum(i["quantity"] * i["unitPrice"] for i in order_items)
#     order = {
#         "_id": ObjectId(),
#         "userId": user_id,
#         "items": order_items,
#         "totalAmount": round(total, 2),
#         "status": random.choice(["pending", "shipped", "delivered", "cancelled"]),
#         "orderedAt": datetime.now() - timedelta(days=random.randint(0, 90))
#     }
#     db.orders.insert_one(order)
#     order_ids.append(order["_id"])

#     payment = {
#         "_id": ObjectId(),
#         "orderId": order["_id"],
#         "userId": user_id,
#         "method": random.choice(["credit_card", "upi", "paypal"]),
#         "amount": order["totalAmount"],
#         "status": "paid",
#         "paidAt": order["orderedAt"] + timedelta(hours=random.randint(1, 48))
#     }
#     db.payments.insert_one(payment)

# # --- Reviews ---
# for _ in range(100):
#     review = {
#         "_id": ObjectId(),
#         "userId": random.choice(user_ids),
#         "productId": random.choice(product_ids),
#         "rating": random.randint(1, 5),
#         "comment": fake.sentence(),
#         "createdAt": datetime.now() - timedelta(days=random.randint(0, 90))
#     }
#     db.reviews.insert_one(review)

# # --- Wishlists ---
# for user_id in user_ids:
#     wishlist = {
#         "_id": ObjectId(),
#         "userId": user_id,
#         "productIds": random.sample(product_ids, random.randint(1, 5)),
#         "updatedAt": datetime.now()
#     }
#     db.wishlists.insert_one(wishlist)

# "✅ All dummy data for e-commerce collections seeded successfully."



# import json
# import random
# from faker import Faker
# from datetime import datetime, timedelta
# from pathlib import Path

# fake = Faker()

# # Sample schema for a realistic hospital application
# hospital_data = {
#     "hospitals": [],
#     "doctors": [],
#     "patients": [],
#     "appointments": [],
#     "tests": [],
#     "operations": [],
#     "bills": []
# }

# specialties = ["Cardiologist", "Neurologist", "Orthopedic", "Pediatrician", "Dermatologist", "General Surgeon", "ENT", "Gynecologist"]
# tests = ["MRI", "Blood Test", "X-Ray", "ECG", "Ultrasound", "CT Scan", "Liver Function Test", "Kidney Function Test"]
# operations = ["Appendectomy", "Angioplasty", "Cataract Surgery", "Gallbladder Removal", "Spine Surgery", "Heart Bypass", "Knee Replacement"]

# # Generate hospitals
# for i in range(5):
#     hospital = {
#         "hospital_id": f"HOSP{i+1}",
#         "name": fake.company() + " Hospital",
#         "location": fake.address(),
#         "contact": fake.phone_number()
#     }
#     hospital_data["hospitals"].append(hospital)

# # Generate doctors
# for i in range(100):
#     doctor = {
#         "doctor_id": f"DOC{i+1}",
#         "name": fake.name(),
#         "specialty": random.choice(specialties),
#         "hospital_id": random.choice(hospital_data["hospitals"])["hospital_id"],
#         "fee": random.randint(300, 1000),
#         "availability": random.choices(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], k=3),
#         "contact": fake.phone_number()
#     }
#     hospital_data["doctors"].append(doctor)

# # Generate patients
# for i in range(300):
#     patient = {
#         "patient_id": f"PAT{i+1}",
#         "name": fake.name(),
#         "dob": str(fake.date_of_birth(minimum_age=1, maximum_age=90)),
#         "gender": random.choice(["Male", "Female"]),
#         "contact": fake.phone_number(),
#         "address": fake.address()
#     }
#     hospital_data["patients"].append(patient)

# # Generate appointments
# for i in range(500):
#     appointment = {
#         "appointment_id": f"APT{i+1}",
#         "patient_id": random.choice(hospital_data["patients"])["patient_id"],
#         "doctor_id": random.choice(hospital_data["doctors"])["doctor_id"],
#         "date": str(fake.date_between(start_date='-1y', end_date='today')),
#         "time": fake.time(),
#         "status": random.choice(["Scheduled", "Completed", "Cancelled"])
#     }
#     hospital_data["appointments"].append(appointment)

# # Generate tests
# for i in range(400):
#     test = {
#         "test_id": f"TEST{i+1}",
#         "name": random.choice(tests),
#         "price": random.randint(200, 5000),
#         "duration_mins": random.randint(10, 90)
#     }
#     hospital_data["tests"].append(test)

# # Generate operations
# for i in range(200):
#     operation = {
#         "operation_id": f"OPR{i+1}",
#         "name": random.choice(operations),
#         "cost": random.randint(15000, 200000),
#         "hospital_id": random.choice(hospital_data["hospitals"])["hospital_id"],
#         "doctor_id": random.choice(hospital_data["doctors"])["doctor_id"]
#     }
#     hospital_data["operations"].append(operation)

# # Generate bills
# for i in range(300):
#     bill = {
#         "bill_id": f"BILL{i+1}",
#         "patient_id": random.choice(hospital_data["patients"])["patient_id"],
#         "amount": random.randint(1000, 150000),
#         "date": str(fake.date_between(start_date='-1y', end_date='today')),
#         "paid": random.choice([True, False])
#     }
#     hospital_data["bills"].append(bill)

# # Save to JSON file
# output_path = Path("/mnt/data/full_hospital_data.json")
# with open(output_path, "w") as f:
#     json.dump(hospital_data, f, indent=2)

# output_path



import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Connect to MongoDB
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["sample_mflix"]




# Load JSON data
with open("D:/personal project/vectorSearch/rag/full_hospital_data.json", "r") as f:
    data = json.load(f)

# Insert into each collection
for collection_name, documents in data.items():
    if isinstance(documents, list):
        print(f"Inserting {len(documents)} into collection: {collection_name}")
        db[collection_name].insert_many(documents)
    else:
        print(f"Skipping {collection_name}, not a list.")

