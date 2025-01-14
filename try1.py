import cohere
import pandas as pd

# Initialize the Cohere client
def initialize_cohere(api_key):
    co = cohere.Client(api_key)  # Initialize Cohere with your API key
    return co

# Function to call Cohere API for generating responses
def get_cohere_response(prompt, co):
    response = co.generate(
        model="command-xlarge",  # Use the appropriate model
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response.generations[0].text.strip()

# Load CSV data and filter for TV products
def load_products(file_path):
    data = pd.read_csv(file_path)
    relevant_columns = [
        "name", "model", "combo", "size", "price", 
        "description", "specification", "features", 
        "sub_category", "category", "parent_category"
    ]
    filtered_data = data[relevant_columns]
    tv_products = filtered_data[filtered_data["parent_category"].str.lower() == "tv"]
    return tv_products

# Format the product details for the LLM
def format_product_details(products):
    return "\n".join(
        f"- Name: {row['name']}, Model: {row['model']}, Combo: {row['combo']}, "
        f"Size: {row['size']}, Price: {row['price']}, Description: {row['description']}"
        for _, row in products.iterrows()
    )

# Filter products and explain suitability based on user description
def filter_products_with_explanations(products, user_description, co):
    prompt_template = f"""
    Based on the user's description: "{user_description}",
    suggest the best TV products from the list below and explain why they are suitable.
    For each suggestion, provide the name, model, size, price, and a brief reason for suitability.

    {format_product_details(products)}

    Provide the top 3 recommendations in the following format:
    - Name: ...
      Model: ...
      Size: ...
      Price: ...
      Suitability: (Explain briefly why this product meets the user's needs)
    """

    response = get_cohere_response(prompt_template, co)
    return response

# Main function
def main():
    file_path = "FinalSony_TV_Audio_Cameras_data.csv"  # Replace with your actual file path
    api_key = "lFPJJUNoM3hmzNVjvvMa1ZkPqzqzBTCccNBX7GU4"  # Replace with your Cohere API key

    # Initialize Cohere
    co = initialize_cohere(api_key)

    # Load TV products from CSV
    products = load_products(file_path)

    # Get user input
    user_description = input("Describe the situation or what you're looking for: ")

    # Get product suggestions with explanations
    suggestions = filter_products_with_explanations(products, user_description, co)
    print("\nTop Recommendations with Suitability:")
    print(suggestions)

if __name__ == "__main__":
    main()
