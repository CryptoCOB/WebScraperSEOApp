from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from imports import *
from utils import generate_content_descriptions, predict_next_character,analyze_and_improve_content


class SEOAnalyzerApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Text input for SEO content with predictive text capabilities
        self.content_input = TextInput(hint_text='Enter your content here...', multiline=True)
        self.content_input.bind(on_text_validate=self.on_text_validate)
        self.layout.add_widget(self.content_input)

        # Button to generate content descriptions
        self.generate_button = Button(text='Generate Descriptions')
        self.generate_button.bind(on_press=self.generate_descriptions)
        self.layout.add_widget(self.generate_button)

        # Button to analyze and improve content
        self.analyze_button = Button(text='Analyze and Improve Content')
        self.analyze_button.bind(on_press=self.analyze_content)
        self.layout.add_widget(self.analyze_button)

        # Label to display results
        self.results_label = Label(text='Results will be displayed here.')
        self.layout.add_widget(self.results_label)

        return self.layout

    def on_text_validate(self, instance):
        """Handle predictive text generation."""
        last_word = self.content_input.text.split(' ')[-1]
        prediction = predict_next_character(last_word)
        self.content_input.text += prediction

    def generate_descriptions(self, instance):
        """Call the function to generate content descriptions."""
        descriptions = generate_content_descriptions(self.content_input.text)
        self.results_label.text = 'Generated Descriptions: ' + descriptions

    def analyze_content(self, instance):
        """Call the function to analyze and improve content."""
        improved_content = analyze_and_improve_content(self.content_input.text)
        self.results_label.text = 'Improved Content: ' + improved_content

    def on_stop(self):
        """Cleanup when the application is about to stop."""
        print("Cleaning up the application...")

if __name__ == "__main__":
    SEOAnalyzerApp().run()
