# STEP 1: Load modules
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# STEP 2: Load Model and Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

# STEP 3: Encodin
input_text = "인천에 위치한 대형 골프장에서 캐디로 일하고 있는 50대 권종희 씨는 경력 20년이 되어가는 지금도 고객을 상대하는 일이 즐겁다. 제각기 다른 환경에서 살아온 손님들이 들려주는 이야기는 흥미롭고 인생의 교훈도 얻을 수 있기 때문이다.마음 같아서는 몸이 허락하는 한 일을 계속하고 싶지만, 현실은 녹록지 않다. 중장년의 캐디에게 서비스를 받는 것을 부담스러워하는 고객들이 많아서다. 사업장은 60세까지 정년을 보장하고 있지만, 머지않아 골프장을 떠나야 할 수도 있겠다는 생각에 권 씨는 때로 씁쓸함을 느낀다.권 씨의 고민은 자연스레 노후 생활로 이어진다. 현재 그가 국민연금 보험료를 납부한 기간은 20대 시절 회사 생활을 포함해도 10년 남짓. 정년까지 납부해도 20년을 채우기 어렵다. 현행 국민연금제도가 40년간 보험료를 납부해야 소득대체율의 40%를 보장한다는 점을 고려하면, 추후 권 씨가 받을 연금으로는 일상생활을 유지하기 어렵다는 결론이 나온다.30여 년간 일해왔음에도 국민연금 가입 기간이 턱없이 짧은 이유는 단 하나, 삶이 빠듯해 소득의 9%에 달하는 보험료를 지불하기 부담스러웠기 때문이다. 일반 직장가입자는 보험료 중 절반만 부담하면 나머지 절반은 사업체가 대신 내주지만, '특수형태근로종사자(특고)'로 분류된 캐디는 지역가입자로 분류돼 보험료를 전액 부담해야 한다.권 씨는 국민연금의 중요성을 알면서도, 비수기가 되면 소득이 크게 줄어 보험료를 견디기 어려웠다라며 우리(캐디)도 일반 직장인들처럼 사업장이 보험료를 분담해 주면, 골프장에 소속감이 생겨 좀 더 열정적으로 일할 수 있을 것 같다고 했다."
inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)

# STEP 4: Run Inference (Generate Summary Text Ids)
summary_text_ids = model.generate(
input_ids=inputs['input_ids'],
attention_mask=inputs['attention_mask'],
bos_token_id=model.config.bos_token_id,
eos_token_id=model.config.eos_token_id,
length_penalty=1.0,
max_length=300,
min_length=12,
num_beams=6,
repetition_penalty=1.5,
no_repeat_ngram_size=15,
)

# STEP 5: Check results (Decoding Text Ids)
print(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))