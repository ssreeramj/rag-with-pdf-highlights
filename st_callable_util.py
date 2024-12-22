import inspect
import json
import os
from typing import Any, Callable, TypeVar
from urllib.parse import quote, urlencode

from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Define a type variable for generic type hinting in the decorator, to maintain
# the return type of the input function and the wrapped function
fn_return_type = TypeVar("fn_return_type")


def create_viewer_url_by_passage(passage):
    """Create a URL to open PDF.js viewer with annotation highlighting."""

    base_url = "http://localhost:8003/viewer.html"

    try:
        ann_list = json.loads(passage.metadata.get("annotations", "[]"))
        pdf_url = passage.metadata.get("source", None)
        if not pdf_url or not ann_list:
            return None

        # Convert each annotation to include page information
        viewer_annotations = []
        for ann in ann_list:
            viewer_annotations.append(
                {
                    "x": ann.get("x", 0),
                    "y": ann.get("y", 0),
                    "width": ann.get("width", 0),
                    "height": ann.get("height", 0),
                    "page": ann.get(
                        "page", 0
                    ),  # Include the page number for each annotation
                }
            )

        # Create a single URL with all annotations
        params = {
            "file": pdf_url,
            "annotations": json.dumps(viewer_annotations),
            "pageNumber": viewer_annotations[0]["page"]
            + 1,  # Start with first annotated page
        }
        return f"{base_url}?{urlencode(params, quote_via=quote)}"

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error in create_viewer_url_by_passage: {e}")
        return None


# Decorator function to add the Streamlit execution context to a function
def add_streamlit_context(
    fn: Callable[..., fn_return_type],
) -> Callable[..., fn_return_type]:
    """
    Decorator to ensure that the decorated function runs within the Streamlit execution context.

    Args:
        fn (Callable[..., fn_return_type]): The function to be decorated.

    Returns:
        Callable[..., fn_return_type]: The decorated function that includes
        the Streamlit context setup.
    """
    # Retrieve the current Streamlit script execution context.
    # This context holds session information necessary for Streamlit operations.
    ctx = get_script_run_ctx()

    def wrapper(*args, **kwargs) -> fn_return_type:
        """
        Wrapper function that adds the Streamlit context and then calls the original function.

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            fn_return_type: The result from the original function.
        """
        add_script_run_ctx(ctx=ctx)  # Set the correct Streamlit context for execution
        return fn(*args, **kwargs)  # Call the original function with its arguments

    return wrapper


# Define a custom callback handler class for managing and displaying stream
# events from LangGraph in Streamlit
class StreamHandler(BaseCallbackHandler):
    """
    Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
    """

    def __init__(self, container: DeltaGenerator, initial_text: str = ""):
        """
        Initializes the StreamHandler with a Streamlit container and optional initial text.

        Args:
            container (DeltaGenerator): The Streamlit container where text will be rendered.
            initial_text (str): Optional initial text to start with in the container.
        """
        self.container = container  # The Streamlit container to update
        self.token_placeholder = (
            self.container.empty()
        )  # Placeholder for dynamic token updates
        self.text = (
            initial_text  # Initialize the text content, starting with any initial text
        )
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        # print(f"{prompts[0]=}")
        if prompts[0].endswith("Formulate an improved question: "):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Callback method triggered when a new token is received (e.g., from a language model).

        Args:
            token (str): The new token received.
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("run_id") == self.run_id_ignore_token:
            return
        self.text += token  # Append the new token to the existing text
        self.token_placeholder.write(
            self.text
        )  # Update the Streamlit container with the full text


class PrintRetrievalHandler(BaseCallbackHandler):
    """retriever callback class"""

    def __init__(self, container):
        self.status = container.status("**Thinking...**")
        self.retriever_start_check = False

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.retriever_start_check = True
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        # check if we just got a general answer or from the retriever
        if self.retriever_start_check is False:
            self.status.update(label="General Answer", state="complete")

    def on_retriever_end(self, documents, **kwargs):
        self.status.write(f"**File Name: {documents[0].metadata['source']}**")
        for _, doc in enumerate(documents):
            source_url = create_viewer_url_by_passage(passage=doc)
            page_num = doc.metadata["pageNum"]

            self.status.markdown(
                doc.page_content[:200] + f"..............[Page Num: {page_num}]({source_url})"
            )

        self.status.update(state="complete")


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided
    Streamlit container with new tokens.

    Args:
        parent_container (DeltaGenerator): The Streamlit container where
        the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    st_cb = StreamHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith(
            "on_"
        ):  # Identify callback methods that respond to LLM events
            setattr(
                st_cb, method_name, add_streamlit_context(method_func)
            )  # Wrap and replace the method with the context-aware version

    # Return the fully configured StreamlitCallbackHandler instance,
    # now context-aware and integrated with any ChatLLM
    return st_cb


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_retriever_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Retriever callback handler that updates the provided
    Streamlit container with new tokens.

    Args:
        parent_container (DeltaGenerator): The Streamlit container where
        the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    rt_cb = PrintRetrievalHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(
        rt_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith(
            "on_"
        ):  # Identify callback methods that respond to LLM events
            setattr(
                rt_cb, method_name, add_streamlit_context(method_func)
            )  # Wrap and replace the method with the context-aware version

    # Return the fully configured StreamlitCallbackHandler instance,
    # now context-aware and integrated with any ChatLLM
    return rt_cb
