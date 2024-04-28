from imports import *
import database
import asyncio
from config import *
from utils import UserFeedbackCollector, AgentManager, ResourceMonitor
from nltk.corpus import stopwords
from string import punctuation

# AI Model Initialization
MODEL_ID = 'vikhyatk/moondream2'
TOKENIZER, MODEL = load_ai_components()  # Assuming this function is properly defined in utils.py or imports.py

resource_monitor = ResourceMonitor()
feedback_collector = UserFeedbackCollector(database)
agent_manager = AgentManager()

def process_text(text):
    """
    Preprocess text by cleaning and normalizing it for LLM processing.
    """
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in punctuation])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def additional_processing(text):
    """
    Post-process LLM output to refine and enhance the response.
    """
    # Example: capitalize the first letter of each sentence
    text = '. '.join([sentence.capitalize() for sentence in text.split('. ')])
    return text

async def generate_content_descriptions(images):
    descriptions = {}
    async with aiohttp.ClientSession() as session:
        for image_path in images:
            resource_monitor.log_usage()
            try:
                image = Image.open(image_path)
                enc_image = await MODEL.encode_image_async(image)
                description = await MODEL.answer_question_async(enc_image, "Describe this image.")
                descriptions[image_path] = description
                await database.insert_image_description(image_path, description)
                feedback_collector.collect_feedback(image_path, description)
            except Exception as e:
                logging.error(f"Error generating description for {image_path}: {str(e)}")
                descriptions[image_path] = "Description unavailable."
    return descriptions

async def analyze_and_improve_content(text):
    try:
        resource_monitor.log_usage()
        text = process_text(text)
        prompt = f"How can this content be improved for better readability and SEO? {text}"
        inputs = TOKENIZER(prompt, return_tensors='pt')
        outputs = await MODEL.generate_async(**inputs, max_new_tokens=150)
        suggestions = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        suggestions = additional_processing(suggestions)
        feedback_collector.collect_feedback(text, suggestions)
        return suggestions
    except Exception as e:
        logging.error(f"Error analyzing content: {str(e)}")
        return "Analysis unavailable."

def interactive_content_session():
    print("LLM> Welcome to the interactive content session. Type 'exit' to end the session.")
    while True:
        user_input = input("You> ")
        if user_input.lower() == 'exit':
            print("LLM> Ending session.")
            break
        try:
            result = asyncio.run(analyze_and_improve_content(user_input))
            print(f"LLM> {result}")
        except Exception as e:
            logging.error(f"Error in interactive_content_session: {str(e)}")

if __name__ == "__main__":
    agent_manager.add_agent(interactive_content_session)
