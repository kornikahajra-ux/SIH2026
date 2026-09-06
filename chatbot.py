from huggingface_hub import InferenceClient
import xarray as xr

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
    token="hf_YOUR_API_KEY_HERE"
)
system_message = {
    "role": "system",
    "content": """You are Oceanis, an expert oceanographic and geospatial AI assistant. 
    Your goal is to help users analyze marine data, including Sea Surface Temperature (SST), 
    salinity, height, anomaly, winds and ocean currents.

    When asked about a specific location, ALWAYS extract the bounding box or coordinates. 
    If you need to fetch data, reply with a structured JSON tool call requesting the 
    specific dataset (e.g., {"action": "fetch_sst", "lat": 45.0, "lon": -30.0}). 
    
    Do not guess environmental data. Rely on the context provided to you by the backend data tools."""
}
conversation_history = [system_message]

ocean_data = xr.open_dataset("./local_data/sea_surface_temp_2026.nc")

def chat(user_input):
    conversation_history.append({"role": "user", "content": user_input})
    try:
        response = client.chat_completion(
            messages=conversation_history,
            max_tokens=500,
            temperature=0.7
        )
        assistant_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply
    except Exception as e:
        return f"Error connecting to Hugging Face: {e}"

user_input = input("Enter query:")
reply = chat(user_input)
print("Bot:",reply)
