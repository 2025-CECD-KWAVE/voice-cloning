import os, sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import traceback,torchaudio
import re, json
import torch
from text.LangSegmenter import LangSegmenter
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn,SynthesizerTrnV3
import numpy as np
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime

model_version = "v2"
sovits_path = "SoVITS_weights_v2/IUv2_e8_s216.pth"
gpt_path = "GPT_weights_v2/IUv2-e15.ckpt"

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
    
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)
    
resample_transform_dict={}
def resample(audio_tensor, sr0):
    global resample_transform_dict
    if sr0 not in resample_transform_dict:
        resample_transform_dict[sr0] = torchaudio.transforms.Resample(
            sr0, 24000
        ).to(device)
    return resample_transform_dict[sr0](audio_tensor)
    
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
          
def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError("올바른 텍스트를 입력하십시오")
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text
  
def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts
  
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)
  
def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

from module.mel_processing import spectrogram_torch,mel_spectrogram_torch
spec_min = -12
spec_max = 2
def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn=lambda x: mel_spectrogram_torch(x, **{
    "n_fft": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "num_mels": 100,
    "sampling_rate": 24000,
    "fmin": 0,
    "fmax": None,
    "center": False
})

def get_spepc(hps, filename):
    # audio = load_audio(filename, int(hps.data.sampling_rate))
    audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    language = language.replace("all_","")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text
  
def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T
  
dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert
  
from text import chinese
def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text

from process_ckpt import get_sovits_version_from_path_fast,load_sovits_new
dict_language_v2 = {
    "중국어": "all_zh",#全部按中文识别
    "영어": "en",#全部按英文识别#######不变
    "일본어": "all_ja",#全部按日文识别
    "광둥어": "all_yue",#全部按中文识别
    "한국어": "all_ko",#全部按韩文识别
    "중영혼합": "zh",#按中英混合识别####不变
    "일영혼합": "ja",#按日英混合识别####不变
    "광영혼합": "yue",#按粤英混合识别####不变
    "한영혼합": "ko",#按韩英混合识别####不变
    "다국어혼합": "auto",#多语种启动切分识别语种
    "다국어혼합(광둥)": "auto_yue",#多语种启动切分识别语种
}
def change_sovits_weights(sovits_path,prompt_language=None,text_language=None):
    global vq_model, hps, version, model_version, dict_language,if_lora_v3
    version, model_version, if_lora_v3=get_sovits_version_from_path_fast(sovits_path)
    # print(sovits_path,version, model_version, if_lora_v3)
    """ V3
    if if_lora_v3==True and is_exist_s2gv3==False:
        info= "GPT_SoVITS/pretrained_models/s2Gv3.pth" + "SoVITS V3 기본 틀이 없습니다.")
        gr.Warning(info)
        raise FileExistsError(info)
    """
    #dict_language = dict_language_v1 if version =='v1' else dict_language_v2
    dict_language = dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = {'__type__':'update'}, {'__type__':'update', 'value':prompt_language}
        else:
            prompt_text_update = {'__type__':'update', 'value':''}
            prompt_language_update = {'__type__':'update', 'value':"중국어"}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
        else:
            text_update = {'__type__':'update', 'value':''}
            text_language_update = {'__type__':'update', 'value':"중국어"}
        if model_version=="v3":
            visible_sample_steps=True
            visible_inp_refs=False
        else:
            visible_sample_steps=False
            visible_inp_refs=True
        yield  {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update,{"__type__": "update", "visible": visible_sample_steps},{"__type__": "update", "visible": visible_inp_refs},{"__type__": "update", "value": False,"interactive":True if model_version!="v3"else False},{"__type__": "update", "visible":True if model_version=="v3"else False}

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if 'enc_p.text_embedding.weight'not in dict_s2['weight']:
        hps.model.version = "v2"#v3model,v2sybomls
    elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version=hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version!="v3":
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        model_version=version
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
    if ("pretrained" not in sovits_path):
        try:
            del vq_model.enc_q
        except:pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3==False:
        print("loading sovits_%s"%model_version,vq_model.load_state_dict(dict_s2["weight"], strict=False))
    """ V3
    else:
        print("loading sovits_v3pretrained_G", vq_model.load_state_dict(load_sovits_new(path_sovits_v3)["weight"], strict=False))
        lora_rank=dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_v3_lora%s"%(lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()
        """

    with open("./weight.json")as f:
        data=f.read()
        data=json.loads(data)
        data["SoVITS"][version]=sovits_path
    with open("./weight.json","w")as f:f.write(json.dumps(data))


try:next(change_sovits_weights(sovits_path))
except:pass

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./weight.json")as f:
        data=f.read()
        data=json.loads(data)
        data["GPT"][version]=gpt_path
    with open("./weight.json","w")as f:f.write(json.dumps(data))


change_gpt_weights(gpt_path)

now_dir = os.getcwd()
def init_bigvgan():
    global bigvgan_model
    from BigVGAN import bigvgan
    bigvgan_model = bigvgan.BigVGAN.from_pretrained("%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,), use_cuda_kernel=False)  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)

if model_version!="v3":bigvgan_model=None
else:init_bigvgan()

sr_model=None
def audio_sr(audio,sr):
    global sr_model
    if sr_model==None:
        from tools.audio_sr import AP_BWE
        try:
            sr_model=AP_BWE(device,DictToAttrRecursive)
        except FileNotFoundError:
            gr.Warning("당신은 초과 점수 모델의 매개변수를 다운로드하지 않았으므로 초과 점수를 진행하지 않습니다. 점수를 초과하려면 먼저 튜토리얼을 참조하여 파일을 다운로드하십시오")
            return audio.cpu().detach().numpy(),sr
    return sr_model(audio,sr)

cache= {}
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="자르지 않음", top_k=20, top_p=0.6, temperature=0.6, ref_free = False,speed=1,if_freeze=False,inp_refs=None,sample_steps=8,if_sr=False,pause_second=0.3):
    global cache
    if ref_wav_path:pass
    else:gr.Warning('레퍼오디오 필요')
    if text:pass
    else:gr.Warning('추론 텍스트 필요')
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    if model_version=="v3":
        ref_free=False#s2v3暂不支持ref_free
    else:
        if_sr=False
    t0 = ttime()
    prompt_language = dict_language_v2[prompt_language]
    text_language = dict_language_v2[text_language]


    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print("실제 입력된 참고 텍스트:", prompt_text)
    text = text.strip("\n")
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print("실제 입력된 참고 텍스트:", text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                gr.Warning("참고 오디오는 3~10초여야 함")
                raise OSError("참고 오디오는 3~10초여야 함")
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1-t0)

    text = cut1(text)
    """
    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    """
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print("실제 입력 대상 텍스트 (절구 후)", text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    ###s2v3暂不支持ref_free
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print("실제 입력한 대상 텍스트 (문장당):", text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
        print("프런트엔드 처리 후 텍스트 (문장마다):", norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text]=pred_semantic
        t3 = ttime()
        ###v3不存在以下逻辑和inp_refs
        if model_version!="v3":
            refers=[]
            if(inp_refs):
                for path in inp_refs:
                    try:
                        refer = get_spepc(hps, path.name).to(dtype).to(device)
                        refers.append(refer)
                    except:
                        traceback.print_exc()
            if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
            audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed)[0][0]#.cpu().detach().numpy()
        else:
            refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)
            phoneme_ids0=torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1=torch.LongTensor(phones2).to(device).unsqueeze(0)
            # print(11111111, phoneme_ids0, phoneme_ids1)
            fea_ref,ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio=ref_audio.to(device).float()
            if (ref_audio.shape[0] == 2):
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            if sr!=24000:
                ref_audio=resample(ref_audio,sr)
            # print("ref_audio",ref_audio.abs().mean())
            mel2 = mel_fn(ref_audio)
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            if (T_min > 468):
                mel2 = mel2[:, :, -468:]
                fea_ref = fea_ref[:, :, -468:]
                T_min = 468
            chunk_len = 934 - T_min
            # print("fea_ref",fea_ref,fea_ref.shape)
            # print("mel2",mel2)
            mel2=mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge,speed)
            # print("fea_todo",fea_todo)
            # print("ge",ge.abs().mean())
            cfm_resss = []
            idx = 0
            while (1):
                fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                if (fea_todo_chunk.shape[-1] == 0): break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                # set_seed(123)
                cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
                cfm_res = cfm_res[:, :, mel2.shape[2]:]
                mel2 = cfm_res[:, :, -T_min:]
                # print("fea", fea)
                # print("mel2in", mel2)
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            cmf_res = torch.cat(cfm_resss, 2)
            cmf_res = denorm_spec(cmf_res)
            if bigvgan_model==None:init_bigvgan()
            with torch.inference_mode():
                wav_gen = bigvgan_model(cmf_res)
                audio=wav_gen[0][0]#.cpu().detach().numpy()
        max_audio=torch.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)#zero_wav
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt=torch.cat(audio_opt, 0)#np.concatenate
    sr=hps.data.sampling_rate if model_version!="v3"else 24000
    if if_sr==True and sr==24000:
        print("오디오 과점 중")
        audio_opt,sr=audio_sr(audio_opt.unsqueeze(0),sr)
        max_audio=np.abs(audio_opt).max()
        if max_audio > 1: audio_opt /= max_audio
    else:
        audio_opt=audio_opt.cpu().detach().numpy()
    return sr, (audio_opt * 32767).astype(np.int16)