from datetime import datetime
from boring_semantic_layer import from_yaml

models = from_yaml(
    './data/northwind_bsl.yaml', 
    profile='northwind_duckdb',
    profile_path='./data/northwind_profile.yaml')

print("=== Models")
for k, v in models.items():
    print(f"{v}\n")

print("=== Queries")
print("Customer count per country:")
print(
    models['Customers'] \
    .group_by('Country') \
    .aggregate('CustomerCount') \
    .execute())
print()

print("Product quantity per country:")
print(
    models['Products'] \
    .group_by('Country') \
    .aggregate('Products.ProductCount').execute())
print()

print("Sales per category in 2016:")
print(
    models['OrderDetails'] \
    .filter(lambda t: t.OrderDate.year() == 2016) \
    .group_by('CategoryName') \
    .aggregate('OrderDetails.TotalRevenue').execute())
print()
