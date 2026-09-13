# Tech Challenge Fase 4 — Saúde da Mulher (Multimodal)

Solução de monitoramento multimodal (vídeo + áudio + texto) para identificação precoce de riscos em saúde e segurança feminina.

## Funcionalidades cobertas

1. **Análise de vídeo clínica** — YOLOv8 customizado (`bleeding`), DeepFace/Haar (emoções) e MediaPipe Pose
2. **Análise de áudio de consultas** — MoviePy + SpeechRecognition (+ Whisper API opcional) e scores de risco (PPD, ansiedade, violência, fadiga)
3. **Nuvem** — AWS Comprehend + OpenAI GPT (com fallbacks locais sem credenciais)

## Objetivos cobertos

- Detecção precoce de riscos maternos/ginecológicos (sangramento visual)
- Sinais de violência doméstica / abuso
- Bem-estar psicológico feminino
- Serviços em nuvem para NLP/relatório
- Alertas preventivos por fusão multimodal

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # opcional: OPENAI_API_KEY / AWS_*
```

Opcional (mais pesado, alinhado às aulas):

```bash
pip install deepface tf-keras
```

## Preparar dados demo + treinar YOLO

```bash
python scripts/prepare_sample_data.py
python scripts/train_yolo.py
```

Sem treino, o detector usa **fallback HSV** para regiões vermelhas (ainda gera alertas na demo).

## Executar pipeline

```bash
# Vídeo + texto de pós-parto
python -m src.main \
  --video data/samples/consulta_com_sangramento.mp4 \
  --text-file data/samples/consulta_pos_parto.txt \
  --transcript-override "Estou triste e chorando, não consigo dormir" \
  --prefix demo_pos_parto

# Triagem de violência
python -m src.main \
  --video data/samples/consulta_sem_sangramento.mp4 \
  --text-file data/samples/triagem_violencia.txt \
  --audio data/samples/consulta_hesitante.wav \
  --transcript-override "Tenho medo dele e ele me bate" \
  --prefix demo_violencia
```

Saídas em `outputs/` (JSON + Markdown + vídeo anotado).

## Testes

```bash
python -m pytest tests/ -q
# ou sem pytest:
python -m unittest tests.test_fusion_rules
```

## Estrutura

- `src/video/` — YOLOv8, emoção, pose
- `src/audio/` — transcrição e risco vocal
- `src/text/` — classificação, sumarização, Comprehend/OpenAI
- `src/fusion/` — fusão e alertas
- `RELATORIO_TECNICO.md` — entrega acadêmica

## Entrega acadêmica

- Código: este repositório
- Relatório: [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md)
- Exemplos de saída: `outputs/examples/`
- Métricas YOLO: `models/bleeding_metrics.json`
- Vídeo YouTube/Vimeo: fora do escopo desta entrega

## Privacidade

Os dados em `data/samples` são **sintéticos**. Não use PHI real sem consentimento e controles adequados (LGPD).
