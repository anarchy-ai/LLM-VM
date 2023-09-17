import gradio as gr # pylint: disable=import-error

from llm_vm.client import Client

client = Client(big_model='chat_gpt', small_model='pythia')


def anarchy_client(prompt, context, openai_key=None, temperature=0.0, data_synthesis=False, finetune=False):
    response = client.complete(
        prompt=prompt,
        context=context,
        openai_key=openai_key,
        temperature=temperature,
        data_synthesis=data_synthesis,
        finetune=finetune,
    )

    return response


interface = gr.Interface(
    fn=anarchy_client,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter prompt here..."),
        gr.Textbox(lines=2, placeholder="Context, if there's any"),
        gr.Text(placeholder="Open API key"),
        gr.Slider(0, 1),
        gr.Checkbox(),
        gr.Checkbox(),
    ],
    outputs=gr.Text(),
)

interface.launch()
