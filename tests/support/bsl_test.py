# BSL model test queries
from boring_semantic_layer import from_yaml
from datetime import datetime

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

print("Sales in 2016 by category:")
print(
    models['OrderDetails'].query(
        dimensions=['Categories.CategoryName', 'Orders.OrderDate'],
        measures=['OrderDetails.TotalRevenue'],
        time_grain='TIME_GRAIN_YEAR',
        time_range={'start': '2016-01-01T00:00:00Z', 'end': '2016-12-31T23:59:59Z'},
        order_by=[('OrderDetails.TotalRevenue', 'desc')],
    ).execute())
print()

print("Sales per category in 2016:")
print(
    models['OrderDetails'] \
    .filter(lambda t: t.OrderDate.year() == 2016) \
    .group_by('CategoryName') \
    .aggregate('OrderDetails.TotalRevenue').execute())
print()
