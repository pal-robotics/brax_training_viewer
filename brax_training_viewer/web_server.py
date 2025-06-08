import uvicorn

def run_web_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """
    Launches the FastAPI web server for visualization.
    """
    uvicorn.run("brax_training_viewer.backend:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    run_web_server() 