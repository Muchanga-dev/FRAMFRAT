import os
import subprocess

def run_streamlit():
    """
    Executa o aplicativo Streamlit programaticamente.
    """
    try:
        # Define o comando para iniciar o Streamlit
        cmd = ["streamlit", "run", "app.py"]
        # Executa o comando e redireciona os fluxos de entrada/saída
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("Iniciando o FRAMFRAT...")
        for line in process.stdout:
            print(line.decode("utf-8").strip())

        process.wait()
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"Erro ao executar o Streamlit: {e}")

if __name__ == "__main__":
    # Verifica se o arquivo app.py existe antes de iniciar
    if not os.path.exists("app.py"):
        print("Erro: O arquivo app.py não foi encontrado no diretório atual.")
    else:
        run_streamlit()
