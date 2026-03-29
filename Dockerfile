FROM julia:1.10-bookworm

# 2. Instalacja Pythona i niezbędnych zależności systemowych
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalacja bibliotek Python (z flagą dla nowszych wersji Debiana)
# Dodajemy jupyterlab i ipykernel, aby VS Code mógł się podpiąć
RUN pip3 install \
    pysr \
    pandas \
    numpy \
    jupyter \
    ipykernel \
    matplotlib \
    scipy \
    lmfit \
    gplearn \
    --break-system-packages

# 4. Konfiguracja silnika Julii i PySR
# Instalujemy pakiety bezpośrednio w systemowej Julii kontenera
RUN julia -e 'using Pkg; Pkg.add(["SymbolicRegression", "Serialization", "PythonCall"]); Pkg.precompile()'

# 5. Ustawienie folderu roboczego
WORKDIR /workspaces/app

# 6. Informacja o portach (8888 dla Jupytera)
EXPOSE 8888

# Domyślny start - powłoka bash (VS Code sam przejmie kontrolę)
CMD ["/bin/bash"]