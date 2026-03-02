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

print("Sales per country:")
print(
    models['OrderDetails'] \
    .group_by('Country') \
    .aggregate('OrderDetails.TotalNetSales').execute())
print()
