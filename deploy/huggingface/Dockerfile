# Hugging Face Spaces image (copied to the Space root by deploy_hf.py).
# Python 3.12, not 3.14 like Streamlit Cloud: Cellpose 3 pins numpy<2.1,
# which has no wheels for 3.14. With 16 GB on a free Space, Cellpose runs
# online next to the classical methods and the U-Net.
FROM python:3.12-slim

# Spaces run the container as user 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# requirements.txt points pip at the CPU-only PyTorch wheels; installing
# Cellpose in the same command keeps it on those wheels too
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt "cellpose>=3.1,<4"

COPY --chown=user . .
USER user
ENV HOME=/home/user MPLCONFIGDIR=/home/user/.cache/matplotlib

# Download Cellpose's nuclei model at build time and check its pinned SHA-256,
# so visitors never wait for (or depend on) cellpose.org
RUN python -c "from src.segmentation import _load_verified_cellpose; _load_verified_cellpose()"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# Spaces show the app inside an iframe on huggingface.co, where Streamlit's
# XSRF cookie is blocked and every upload fails with 403. The app has no
# accounts or stored state, so turning the check off exposes nothing.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
