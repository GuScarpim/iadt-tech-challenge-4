# Explicação do Projeto e do Tech Challenge

Documento de leitura rápida: o que o enunciado pede, o que este repositório faz, se está correto, por que cada escolha foi feita e de onde veio cada peça (código, dados, aulas).

---

## 1. O que é o Tech Challenge (Fase 4)

É o projeto integrador da pós **IA para Devs (FIAP)**, valendo grande parte da nota da fase. O cenário é uma rede hospitalar de **saúde da mulher** que quer monitorar pacientes com dados **multimodais** (vídeo, áudio e texto) para detectar riscos cedo e alertar a equipe.

### O enunciado exige, em resumo

| Exigência | Obrigatório? |
|---|---|
| Escolher **≥ 2** funcionalidades (vídeo, áudio, sinais vitais, nuvem Azure etc.) | Sim |
| Escolher **≥ 3** objetivos (risco materno, violência, bem-estar, nuvem, anomalias) | Sim |
| **Análise de vídeo** especializada + **YOLOv8 customizado** (um alvo) | Sim |
| **Análise de áudio** especializada (consulta / pré-natal / pós-parto / trauma) | Sim |
| Relatórios/alertas automáticos | Sim |
| Repositório Git com **código** + **relatório técnico** | Sim |
| Vídeo demo YouTube/Vimeo (até 15 min) | Pedido no PDF; **fora do escopo desta entrega** (decisão do grupo) |

### O que as aulas da fase ensinam (contexto)

As disciplinas da fase 4 cobrem:

1. **Análise de Vídeo, Áudio e Texto** — OpenCV, DeepFace, MediaPipe, MoviePy, SpeechRecognition, classificação e sumarização de texto  
2. **Textract + AWS Comprehend** — extração e NLP na AWS  
3. **Open API** — OpenAI GPT / Whisper  

**Gap importante:** as aulas **não** ensinam YOLOv8 nem Azure. O enunciado exige YOLOv8 e só cita Azure como *opção* de nuvem.

---

## 2. O que este projeto é

Um **pipeline multimodal em Python** que:

1. Analisa um **vídeo** (sangramento com YOLOv8, emoções, pose)  
2. Analisa **áudio/texto** de consulta (transcrição + risco clínico)  
3. Opcionalmente usa **nuvem** (OpenAI + AWS Comprehend)  
4. **Fund e** tudo em scores e gera **alertas** + relatório em `outputs/`

Entrada típica (CLI):

```bash
python -m src.main --video ... --audio ... --text-file ... --prefix demo_xyz
```

Saída: JSON + Markdown com nível `INFO` / `ATENCAO` / `CRITICO` e tipos de alerta (sangramento, PPD, violência etc.).

---

## 3. Opções escolhidas (e por quê)

### Funcionalidades (≥ 2) — escolhemos 3

| Funcionalidade | Por quê |
|---|---|
| Análise de **vídeo** clínico | Obrigatória no enunciado |
| Análise de **áudio** de consulta | Obrigatória no enunciado |
| **Nuvem** (OpenAI + AWS Comprehend) | Fecha o mínimo de “serviços gerenciados”; alinhada às **aulas** (em vez de Azure) |

### Objetivos (≥ 3) — cobrimos os 5

| Objetivo | Como aparece no código |
|---|---|
| Risco materno/ginecológico | YOLOv8 classe `bleeding` |
| Violência / abuso | Léxico + fusão multimodal |
| Bem-estar psicológico | Emoções + PPD/ansiedade |
| Nuvem | Comprehend + GPT |
| Alertas preventivos | Motor em `src/fusion/multimodal_engine.py` |

### YOLOv8 — alvo escolhido

**Sangramento anômalo** (`bleeding`).

Motivos:

- É uma das opções explícitas do PDF  
- Dá para **demonstrar** com dados sintéticos (manchas vermelhas) sem precisar de vídeo cirúrgico real  
- Gera alerta clínico claro: `complicacao_cirurgica_visual`

### Nuvem: AWS/OpenAI em vez de Azure

O PDF menciona Azure como *exemplo*. As aulas usam **AWS + OpenAI**. Mantivemos o que a fase ensina; a funcionalidade “nuvem” continua atendida.

### Vídeo YouTube

**Não entregue nesta versão** — pedido explícito de deixar o vídeo de fora. O restante (código + relatório) cobre a entrega Git.

---

## 4. Está tudo correto?

### Em relação ao enunciado (entrega Git, sem o vídeo demo)

| Critério | Status | Comentário |
|---|---|---|
| ≥ 2 funcionalidades | OK | Vídeo + áudio + nuvem |
| ≥ 3 objetivos | OK | 5 cobertos |
| Vídeo especializado | OK | YOLO + emoção + pose |
| YOLOv8 customizado | OK | Treinado em `bleeding`; pesos em `models/` |
| Áudio especializado | OK | STT + léxico PPD/ansiedade/violência/fadiga |
| Alertas/relatórios | OK | JSON/MD + template clínico |
| Código no Git | OK | Estrutura em `src/` |
| Relatório técnico | OK | `RELATORIO_TECNICO.md` |
| Vídeo YouTube/Vimeo | Fora do escopo | Documentado de propósito |

### Em relação às aulas

| Conteúdo da aula | Onde entrou |
|---|---|
| Emoções / facial | `src/video/emotion_analyzer.py` |
| Pose / ações | `src/video/pose_analyzer.py` |
| Transcrição áudio | `src/audio/transcriber.py` |
| Classificação de texto | `src/text/classifier.py` |
| Sumarização / GPT | `src/text/summarizer.py`, `cloud_nlp.py` |
| AWS Comprehend | `src/text/cloud_nlp.py` |
| YOLOv8 (fora da aula) | `src/video/yolo_detector.py` + `scripts/train_yolo.py` |

### Limitações honestas (ainda assim válidas para o challenge)

- **Dados são sintéticos** — não são exames/consultas reais (bom para LGPD; limitado clinicamente)  
- Áudio demo é tom + silêncio, não fala humana → demos usam `--transcript-override`  
- DeepFace / MediaPipe “clássico” podem cair em **fallback** OpenCV conforme o ambiente  
- Sem chaves AWS/OpenAI, a nuvem usa **fallback local** (o fluxo continua funcionando)  
- Dataset YOLO pequeno e artificial → métricas altas no val sintético **não** equivalem a performance clínica real  

**Conclusão:** para o Tech Challenge (código + relatório + critérios técnicos do PDF, ignorando o vídeo YouTube), a entrega está **completa e alinhada**. Não é um produto hospitalar pronto para produção.

---

## 5. O que foi feito (mapa do repositório)

```text
iadt-tech-challenge-4/
├── README.md                 → como instalar e rodar
├── RELATORIO_TECNICO.md      → entrega acadêmica formal
├── EXPLICACAO_PROJETO.md     → este arquivo (visão geral)
├── requirements.txt
├── .env.example
├── src/
│   ├── main.py               → CLI multimodal
│   ├── config.py             → pesos, limiares, léxico
│   ├── video/                → YOLO, emoção, pose, pipeline
│   ├── audio/                → transcrição + risco vocal
│   ├── text/                 → classificador, sumário, nuvem
│   ├── fusion/               → fusão e alertas
│   └── reporting/            → salva JSON/MD
├── scripts/
│   ├── prepare_sample_data.py → gera vídeos/textos/dataset
│   └── train_yolo.py          → treina YOLOv8n bleeding
├── data/
│   ├── samples/              → demos (vídeo, áudio, texto)
│   └── yolo_bleeding/        → dataset de treino YOLO
├── models/                   → pesos + métricas
├── outputs/                  → resultados das execuções
└── tests/                    → regras de fusão/classificação
```

### Fluxo em uma frase

Vídeo/áudio/texto entram → cada modalidade gera scores → fusão decide o nível de alerta → grava relatório em `outputs/`.

---

## 6. De onde veio cada coisa

### Ideia / requisitos

| Item | Origem |
|---|---|
| Tema saúde da mulher, multimodal, YOLOv8, áudio, alertas | PDF do Tech Challenge Fase 4 |
| Stack DeepFace, MediaPipe, MoviePy, SpeechRecognition, Comprehend, OpenAI | Material das aulas (`aulas-fiap-juntas.pdf`) |
| Escolha sangramento + AWS/OpenAI + sem vídeo YouTube | Decisões de escopo do projeto |

### Código

| Peça | Origem |
|---|---|
| Arquitetura modular `src/*` | Implementação do projeto (não é template FIAP) |
| Padrões de emoção/pose/STT | Inspirados nos hands-on das aulas |
| YOLOv8 Ultralytics | Exigência do enunciado (gap das aulas) |
| Fallbacks (HSV, OpenCV, léxico local) | Para rodar sem GPU/credenciais/libs pesadas |

### Dados (importante)

**Nada veio de hospital, paciente real ou YouTube clínico.**

| Arquivo | Como foi criado |
|---|---|
| `consulta_com_sangramento.mp4` / `consulta_sem_sangramento.mp4` | OpenCV em `prepare_sample_data.py` — cenas artificiais |
| Imagens `data/yolo_bleeding/` | Mesmo script — elipses/manchas vermelhas + labels YOLO |
| `consulta_hesitante.wav` | Tom senoidal + pausas (simula hesitação; **não é voz**) |
| `consulta_pos_parto.txt`, `triagem_violencia.txt`, etc. | Textos **inventados** no estilo de nota clínica |
| Pesos `bleeding_yolov8n.pt` | Treino local com Ultralytics nesse dataset sintético |

Por isso o relatório fala em privacidade / LGPD: a demo não usa PHI.

---

## 7. Por que foi feito assim

1. **Atender o PDF** — vídeo YOLOv8 + áudio + alertas + relatório Git  
2. **Reaproveitar as aulas** — não reinventar stack; encaixar OpenCV/DeepFace/MediaPipe/STT/AWS/OpenAI  
3. **Ser reproduzível** — scripts geram dados e treinam o modelo; CLI documentada no README  
4. **Rodar offline** — fallbacks quando não há API key ou lib opcional  
5. **Evitar dados sensíveis** — sintético em vez de vídeos médicos reais  
6. **Demonstrar fusão** — casos demo (sangramento+PPD, violência, rotina) com níveis de alerta diferentes  

---

## 8. Como validar rapidamente

```bash
source .venv/bin/activate
python -m pytest tests/ -q

python -m src.main \
  --video data/samples/consulta_com_sangramento.mp4 \
  --text-file data/samples/consulta_pos_parto.txt \
  --transcript-override "Estou triste e chorando, não consigo dormir" \
  --prefix demo_pos_parto
```

Esperado: alerta de sangramento e/ou depressão pós-parto; arquivos em `outputs/`.

Exemplos já gerados: `outputs/examples/`.  
Checklist formal: seção 9 de `RELATORIO_TECNICO.md`.

---

## 9. Resposta direta

| Pergunta | Resposta |
|---|---|
| O que é o tech challenge? | Projeto multimodal de saúde da mulher (vídeo+áudio+texto+alertas) |
| O que este repo faz? | Pipeline Python que analisa mídia sintética e gera alertas clínicos demo |
| Está correto? | **Sim** para código + relatório + critérios técnicos (vídeo YouTube de fora) |
| Por que assim? | Enunciado + aulas + dados sintéticos seguros e reproduzíveis |
| De onde veio vídeo/texto? | **Gerados no projeto** (`scripts/prepare_sample_data.py`), não de fontes clínicas reais |

Para a entrega acadêmica detalhada (métricas YOLO, fluxos, checklist), use **`RELATORIO_TECNICO.md`**. Este arquivo é a visão geral em linguagem direta.
