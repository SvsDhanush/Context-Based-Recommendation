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

# Function to call Cohere API for generating responses using `command-r`
def get_cohere_chat_response(prompt, co):
    calculate_token_size(prompt, co)

    response = co.chat(
        model="command-r-plus-08-2024",  # Updated to use the correct model
        messages=[{"role": "user", "content": prompt}]
    )
    # Extract the generated content from the response object
    if response.message and response.message.content:
        # Combine all content items into a single string
        return "\n".join(item.text.strip() for item in response.message.content)
    else:
        raise ValueError("No content generated in the response.")

# Load CSV data
def load_products(file_path):
    data = pd.read_csv(file_path)
    relevant_columns = [
        "name", "model", "combo", "size", "price", 
        "description", "specification", "features", 
        "sub_category", "category", "parent_category"
    ]
    return data[relevant_columns]

# Format product details for the LLM
def format_product_details(products):
    return "\n".join(
        ", ".join([f"{col.capitalize()}: {row[col]}" for col in products.columns])
        for _, row in products.iterrows()
    )

# Filter products and explain suitability based on user description
def filter_products_with_explanations(products, user_description, co):
    product_details = format_product_details(products)
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
    response = get_cohere_chat_response(prompt_template, co)
    return response

# Main function
def main():
    file_path = "FinalSony_TV_Audio_Cameras_data.csv"  # Replace with your actual file path
    api_key = "lFPJJUNoM3hmzNVjvvMa1ZkPqzqzBTCccNBX7GU4"  # Replace with your Cohere API key

    # Initialize Cohere
    co = initialize_cohere(api_key)

    # Load all products
    products = load_products(file_path)

    # Get user description
    user_description = input("Describe what you're looking for in a product: ").strip()

    # Get product suggestions with explanations
    suggestions = filter_products_with_explanations(products, user_description, co)
    print("\nTop Recommendations with Suitability:")
    print(suggestions)

if __name__ == "__main__":
    main()
