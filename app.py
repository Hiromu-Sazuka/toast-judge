import tempfile
import gradio as gr
from PIL import Image

from toast_judge import predict_doneness, labels


def predict(image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)

        score, probs = predict_doneness(tmp.name)
        label = labels[probs.argmax()]

    return (
        label,
        round(float(score), 3)
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.Number(label="Toast Score")
    ],
    examples=[
        ["toast1.jpg"],
        ["toast4.jpg"],
        ["toast10.jpg"],
        ["toast12.jpg"],
    ],
    cache_examples=False,
)

demo.launch()
