import cohere
import pandas as pd

# Initialize the Cohere client
def initialize_cohere(api_key):
    co = cohere.ClientV2(api_key)  # Use ClientV2 for `command-r` model
    return co

# Function to calculate and print token size
def calculate_token_size(prompt, co):
    # Specify the model for tokenization (use "command-r" or another appropriate model)
    tokenized = co.tokenize(text=prompt, model="command-r-plus-08-2024")
    token_size = len(tokenized.tokens)
    print(f"Token size: {token_size}")



# Load CSV data
def load_products(file_path):
    data = pd.read_csv(file_path)
    relevant_columns = [
        "name", "model", "combo", "size", "price", 
        "description", "specification", "features", 
        "sub_category", "category", "parent_category"
    ]
    return data[relevant_columns]

# Filter products based on user-selected category
def filter_by_category(products, category):
    # filtered_products = products[products["parent_category"].str.lower() == category.lower()]
    filtered_products = products
    if filtered_products.empty:
        print(f"No products found in the category '{category}'. Please try again.")
        return None
    return filtered_products

# Format product details based on category
def format_product_details_by_category(products, category):
    # Adjust columns based on category
    if category.lower() == "audio":
        columns_to_include = ["name", "price", "description"]
    elif category.lower() == "camera":
        columns_to_include = ["name", "model", "combo", "price", "description"]
    else:  # Default for TVs
        columns_to_include = ["name", "model", "size", "price", "description"]
    
    # Format products for the LLM
    return "\n".join(
        ", ".join([f"{col.capitalize()}: {row[col]}" for col in columns_to_include])
        for _, row in products.iterrows()
    )

# Filter products and explain suitability based on user description
def filter_products_with_explanations(products, user_description, category, co):
    product_details = format_product_details_by_category(products, category)
    prompt_template = f"""
    Based on the user's description: "{user_description}",
    suggest the best suitable products from the list below and explain why they are suitable.
    For each suggestion, provide the name, model, price, and a brief reason for suitability.

    {product_details}

    Provide the top 3 recommendations in the following format:
    - Name: ...
      Model: ...
      Price: ...
      Suitability: (Explain briefly why this product meets the user's needs)
    """
    response = calculate_token_size(prompt_template, co)
    return response

# Main function
def main():
    file_path = "FinalSony_TV_Audio_Cameras_data.csv"  # Replace with your actual file path
    api_key = "lFPJJUNoM3hmzNVjvvMa1ZkPqzqzBTCccNBX7GU4"  # Replace with your Cohere API key

    # Initialize Cohere
    co = initialize_cohere(api_key)

    # Load all products
    products = load_products(file_path)

    # Ask the user to select a product category
    category = input("Please select a product category (TV, Camera, Audio): ").strip()
    filtered_products = filter_by_category(products, category)

    if filtered_products is None:
        return  # Exit if no products are found in the selected category

    # Get user description
    user_description = input(f"Describe what you're looking for in a {category}: ").strip()

    # Get product suggestions with explanations
    suggestions = filter_products_with_explanations(filtered_products, user_description, category, co)
    print("\nTop Recommendations with Suitability:")
    print(suggestions)

if __name__ == "__main__":
    main()

