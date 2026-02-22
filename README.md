# Python Raytracer

Um raytracer simples, porém poderoso, escrito inteiramente em Python. Este projeto apresenta um motor de raytracing customizado capaz de renderizar várias formas, materiais e condições de iluminação, juntamente com um Editor de Cenas gráfico construído com PySide6.

## Funcionalidades

- **Motor de Raytracing**: Motor customizado com suporte a anti-aliasing, sombras e reflexos.
- **Formas**: Esferas, Cubos, Cilindros e muito mais.
- **Materiais**: Superfícies Foscas (Matte), Xadrez (Checkerboard) e Espelhadas (Mirror).
- **Editor de Cenas**: Uma aplicação GUI (`app.py`) para criar e editar cenas visualmente.
- **Renderizador CLI**: Uma ferramenta de linha de comando (`raster.py`) para renderizar cenas com suporte a multiprocessamento para renderização mais rápida.
- **Definições de Cena**: As cenas são definidas como scripts Python, permitindo a geração programática e complexa de cenas.

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/python-raytracer.git
   cd python-raytracer
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv env
   source env/bin/activate  # No Windows use `env\Scripts\activate`
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Interface de Linha de Comando (CLI)

Você pode renderizar uma cena usando o script `raster.py`. O script recebe o nome do módulo da cena, o número de amostras para anti-aliasing, o número de processos (jobs) para multiprocessamento e o caminho do arquivo de saída.

```bash
python raster.py -s nome_da_cena -n 32 -j 4 -o saida.png
```

**Argumentos:**
- `-s`, `--scene`: O nome do módulo da cena (ex: `ball_scene`, `mirror_scene`). Não inclua a extensão `.py`.
- `-n`, `--num_samples`: Número de amostras por pixel para anti-aliasing (padrão: 32).
- `-j`, `--jobs`: Número de processos paralelos a serem usados (padrão: 4).
- `-o`, `--output`: Caminho para salvar a imagem renderizada (padrão: `output.png`).

**Exemplo:**
```bash
python raster.py -s heart_mitchel_scene -n 64 -j 8 -o imgs/heart_mitchel.png
```

### Editor de Cenas GUI

Para iniciar o editor visual de cenas, execute:

```bash
python app.py
```

O editor permite adicionar objetos, configurar materiais, ajustar a iluminação e visualizar a cena.

## Estrutura do Projeto

- `src/`: Contém o motor principal de raytracing (`base.py`, `camera.py`, `light.py`, `materials.py`, `ray.py`, `shapes.py`, `vector3d.py`).
- `app.py`: O Editor de Cenas GUI em PySide6.
- `raster.py`: O script de renderização via CLI.
- `scene_*.py`: Várias cenas pré-definidas demonstrando as capacidades do motor.
- `docs/`: Documentação e slides em LaTeX.
- `imgs/`: Diretório contendo as imagens renderizadas de saída.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma Issue ou enviar um Pull Request.

## Licença

Este projeto está licenciado sob a Licença MIT.
