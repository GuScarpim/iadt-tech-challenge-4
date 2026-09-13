# Relatório Técnico — Tech Challenge Fase 4

**Curso:** PósTech IA para Devs — FIAP  
**Fase:** 4  
**Tema:** Análise multimodal (vídeo, áudio e texto) para saúde e segurança da mulher  
**Repositório:** código-fonte completo + este relatório  

---

## 1. Contexto e objetivos

A solução monitora pacientes com dados multimodais para identificar sinais precoces de risco em saúde materna/ginecológica, bem-estar psicológico e possíveis situações de violência, gerando alertas para a equipe especializada.

### Funcionalidades implementadas (≥ 2)

| # | Funcionalidade | Status |
|---|---|---|
| 1 | Análise de vídeos clínicos (anomalias / desconforto) | Atendida — YOLOv8 + emoção + pose |
| 2 | Processamento de voz em consultas (PPD, ansiedade, violência, fadiga) | Atendida — STT + léxico + prosódia |
| 3 | Integração com serviços gerenciados em nuvem | Atendida — OpenAI + AWS Comprehend (fallbacks locais) |

### Objetivos atendidos (≥ 3)

| Objetivo | Como |
|---|---|
| Riscos em saúde materna/ginecológica | Detecção de sangramento anômalo (YOLOv8) |
| Sinais de violência doméstica / abuso | Fusão emoção + texto/áudio com léxico de violência |
| Bem-estar psicológico feminino | Emoções visuais + classificação PPD/ansiedade |
| Serviços em nuvem | Comprehend (sentiment/entities) + GPT (relatório) |
| Anomalias / alertas preventivos | Motor de fusão com níveis INFO / ATENCAO / CRITICO |

---

## 2. Fluxo multimodal

```text
[Vídeo]──► YOLOv8 (bleeding) ──┐
         ► Emoção (DeepFace/Haar)─┼──► Fusão ponderada ──► Alertas JSON/MD
         ► Pose (MediaPipe/OpenCV)─┤         │
[Áudio]──► MoviePy + STT ──────────┤         └──► Relatório clínico (OpenAI/local)
         ► Prosódia + risco vocal ─┤
[Texto]──► Classificador clínico ──┤
         ► AWS Comprehend / local ─┘
```

1. **Ingestão** via CLI (`python -m src.main`) com `--video`, `--audio`, `--text` / `--text-file`
2. **Pipelines paralelos** por modalidade
3. **Fusão** com pesos configuráveis em `src/config.py`
4. **Persistência** em `outputs/*.json` e `outputs/*.md`

---

## 3. Modelos por modalidade

### 3.1 Vídeo — YOLOv8 customizado (`bleeding`)

- **Alvo:** sangramento anômalo (uma das opções do enunciado)
- **Base:** Ultralytics YOLOv8n
- **Dataset:** sintético em `data/yolo_bleeding/` (40 train / 10 val), gerado por `scripts/prepare_sample_data.py`
- **Treino:** `python scripts/train_yolo.py` (15 épocas)
- **Métricas de validação** (`models/bleeding_metrics.json`):

| Métrica | Valor |
|---|---|
| Precision | 0.964 |
| Recall | 0.917 |
| mAP50 | 0.915 |
| mAP50-95 | 0.793 |

- **Pesos:** `models/bleeding_yolov8n.pt`
- **Fallback:** máscara HSV vermelha se pesos indisponíveis

### 3.2 Vídeo — emoções (aula 02)

- Preferência: DeepFace (`actions=['emotion']`)
- Fallback: heurística OpenCV (ROI central / Haar quando disponível)
- Score de distresse: soma de `fear`, `sad`, `angry`, `disgust`

### 3.3 Vídeo — pose / fisioterapia (aula 03)

- Preferência: MediaPipe Pose (`mp.solutions`) quando disponível
- Fallback (MediaPipe 1.x sem `solutions`): fluxo óptico OpenCV (movimento + assimetria)
- Indicadores: postura fechada e atividade de membros (proxy de fisioterapia)

### 3.4 Áudio (aula 04)

- Extração: MoviePy 1.x/2.x
- STT: SpeechRecognition; opcional OpenAI Whisper API
- Prosódia: RMS, silêncio/pausas, fadiga
- Classificação clínica da transcrição (léxico supervisionado alinhado à aula 05)

### 3.5 Texto + nuvem (aulas 05–06 + AWS/OpenAI)

- Classificador de risco: `src/text/classifier.py`
- Sumarização / relatório: OpenAI GPT ou template local
- AWS Comprehend: sentiment, entities, key phrases (fallback local sem credenciais)

---

## 4. Fusão e alertas

Pesos (`FUSION_WEIGHTS`):

| Sinal | Peso |
|---|---|
| Sangramento (YOLO) | 0.35 |
| Distresse emocional | 0.20 |
| Postura fechada | 0.10 |
| Risco áudio | 0.25 |
| Risco texto | 0.10 |

Níveis: `INFO` (<0.40), `ATENCAO` (≥0.40), `CRITICO` (≥0.70), elevados também por alertas tipados.

Tipos de alerta:

- `complicacao_cirurgica_visual`
- `desconforto_psicologico_visual`
- `possivel_violencia_domestica`
- `risco_depressao_pos_parto`
- `ansiedade_gestacional`

---

## 5. Resultados e exemplos de anomalias

Arquivos em `outputs/examples/`.

### 5.1 Demo pós-parto + sangramento

Comando:

```bash
python -m src.main \
  --video data/samples/consulta_com_sangramento.mp4 \
  --text-file data/samples/consulta_pos_parto.txt \
  --transcript-override "Estou triste e chorando, não consigo dormir, sem energia" \
  --prefix demo_pos_parto
```

Resultado observado:

- `bleeding` ≈ **0.70** (YOLOv8 detectou frames com sangramento)
- `audio_risk` / `text_risk` = **1.0** (PPD)
- `alert_level` = **CRITICO**
- Alertas: `complicacao_cirurgica_visual`, `risco_depressao_pos_parto`

### 5.2 Demo triagem de violência

```bash
python -m src.main \
  --video data/samples/consulta_sem_sangramento.mp4 \
  --audio data/samples/consulta_hesitante.wav \
  --text-file data/samples/triagem_violencia.txt \
  --transcript-override "Tenho medo dele e ele me bate, não posso falar" \
  --prefix demo_violencia
```

Resultado observado:

- Sem sangramento (`bleeding` = 0)
- Alerta **CRITICO** `possivel_violencia_domestica`

### 5.3 Demo rotina (controle negativo)

```bash
python -m src.main --text-file data/samples/nota_clinica_normal.txt --prefix demo_normal
```

- `alert_level` = **INFO**, sem alertas tipados

### 5.4 Testes automatizados

```bash
python -m pytest tests/ -q
```

**6 passed** — regras de classificação e fusão.

---

## 6. Alinhamento com as aulas da Fase 4

| Aula / disciplina | Uso no projeto |
|---|---|
| Reconhecimento facial / emoções | `emotion_analyzer.py` (DeepFace/Haar) |
| Detecção de atividades / pose | `pose_analyzer.py` (MediaPipe/OpenCV) |
| Transcrição de áudio | `transcriber.py` (MoviePy + SpeechRecognition) |
| Classificação de tópicos | `classifier.py` (risco clínico) |
| Sumarização | `summarizer.py` / relatório GPT |
| AWS Comprehend | `cloud_nlp.py` |
| OpenAI API | Whisper opcional + relatório clínico |
| **Gap do enunciado (YOLOv8)** | Treino customizado Ultralytics |

> O enunciado cita Azure Cognitive Services como *opção*. Optou-se por **OpenAI + AWS Comprehend**, alinhados às aulas da fase, mantendo a funcionalidade de nuvem.

---

## 7. Privacidade e segurança

- Samples em `data/samples` são **sintéticos** (sem PHI real)
- Credenciais apenas via `.env` (não versionado)
- Em produção: consentimento, anonimização, criptografia em trânsito/repouso e controles LGPD

---

## 8. Como reproduzir

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_sample_data.py   # se samples/dataset não existirem
# opcional retreino: python scripts/train_yolo.py
python -m pytest tests/ -q
python -m src.main --video data/samples/consulta_com_sangramento.mp4 \
  --text-file data/samples/consulta_pos_parto.txt \
  --transcript-override "Estou triste e chorando, não consigo dormir" \
  --prefix demo_pos_parto
```

Opcional: `pip install deepface tf-keras` e chaves em `.env` (`OPENAI_API_KEY`, `AWS_*`).

---

## 9. Checklist de validação / revalidação (enunciado)

### Mínimos de escolha

- [x] ≥ 2 funcionalidades (vídeo + áudio + nuvem)
- [x] ≥ 3 objetivos (risco materno, violência, bem-estar, nuvem, alertas)

### Requisitos técnicos obrigatórios

- [x] Análise de vídeo especializada (cirurgia/consulta/fisioterapia/violência — proxies demo)
- [x] YOLOv8 customizado (classe `bleeding` / sangramento anômalo)
- [x] Relatórios/alertas automáticos (JSON + Markdown + template clínico)
- [x] Análise de áudio especializada (PPD, ansiedade, violência, fadiga)

### Entrega Git

- [x] Código-fonte completo
- [x] Relatório técnico (fluxo, modelos, resultados)

### Revalidação pós-implementação

- [x] CLI gera `outputs/` em samples
- [x] YOLO detecta sangramento no vídeo positivo e não no negativo
- [x] Transcript/risco de áudio+texto gera alertas tipados
- [x] Fusão produz alertas não triviais (CRITICO/ATENCAO)
- [x] Testes de regras: 6/6 passando
- [x] Este relatório confere o checklist linha a linha

---

## 10. Limitações e próximos passos

- Dataset YOLO é sintético — para uso clínico real seria necessário corpus anotado por especialistas
- STT em áudio sintético pode falhar; demos usam `--transcript-override` quando necessário
- MediaPipe “clássico” (`solutions`) pode não estar disponível em alguns wheels; há fallback OpenCV
- DeepFace é opcional (dependência pesada)
- Próximos passos: dataset clínico real, calibração de limiares com equipe médica, deploy com autenticação e auditoria
