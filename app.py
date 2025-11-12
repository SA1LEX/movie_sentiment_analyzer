from flask import Flask, render_template, request, jsonify
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== ПРОСТОЙ И ЭФФЕКТИВНЫЙ АНАЛИЗАТОР ====================

def smart_sentiment_analyzer(text):
    """ИСПРАВЛЕННЫЙ анализатор с высокой точностью"""
    text_lower = text.lower().strip()
    
    if not text_lower or len(text_lower) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # СИЛЬНЫЕ ИНДИКАТОРЫ (немедленное определение)
    strong_positive = [
        'великолепен', 'великолепный', 'великолепно', 'шедевр', 'блестящий', 
        'гениальный', 'восхитительный', 'идеальный', 'безупречный', 'незабываемый',
        'потрясающий', 'невероятный', 'фантастический', 'превосходно', 'совершенный',
        'выдающийся', 'прекрасный', 'очаровательный', 'люблю', 'обожаю',
        'восторг', 'восторжен', 'восхищаюсь'
    ]
    
    strong_negative = [
        'хуйня', 'говно', 'пиздец', 'дерьмо', 'отстой', 'мудак', 'гандон', 
        'ублюдок', 'пидорас', 'залупа', 'поебень', 'говнище', 'ебанько',
        'отвратительно', 'отвратительный', 'ужасный', 'кошмар', 'провал',
        'ненавижу', 'бесит', 'раздражает', 'омерзительно'
    ]
    
    # НЕМЕДЛЕННОЕ ОПРЕДЕЛЕНИЕ ПО СИЛЬНЫМ ИНДИКАТОРАМ
    for word in strong_positive:
        if word in text_lower:
            return "ПОЗИТИВНЫЙ", 0.95, "😊"
    
    for word in strong_negative:
        if word in text_lower:
            return "НЕГАТИВНЫЙ", 0.95, "😠"
    
    # БАЗОВЫЕ СЛОВАРЫ
    positive_words = [
        'отличный', 'супер', 'классный', 'шикарный', 'замечательный', 
        'рекомендую', 'советую', 'нравится', 'талантливый', 'мастерство',
        'захватывающий', 'трогательный', 'вдохновляющий', 'глубокий', 
        'профессиональный', 'качественный', 'интересный', 'увлекательный',
        'динамичный', 'симпатичный', 'обаятельный', 'умный', 'оригинальный',
        'свежий', 'новаторский', 'хороший', 'забавный', 'смешной', 'юморной',
        'сильный', 'мощный', 'эпичный', 'красивый', 'эстетичный', 'стильный'
    ]
    
    negative_words = [
        'не рекомендую', 'не советую', 'зря потратил', 'разочарование',
        'разочаровал', 'скучно', 'скучный', 'затянуто', 'предсказуемо',
        'слабый', 'слабая', 'плохой', 'плохая', 'не стоит', 'жалко времени', 
        'жалко денег', 'ожидал большего', 'неудачный', 'неинтересно',
        'банальный', 'шаблонный', 'клишированный', 'примитивный', 'нелогичный',
        'глупый', 'абсурдный', 'нереалистичный', 'фальшивый', 'неестественный',
        'картонный', 'бездушный', 'безвкусный', 'дешевый', 'кустарный',
        'слабовато', 'не очень', 'не совсем', 'сомнительный', 'спорный'
    ]
    
    # КЛЮЧЕВЫЕ ФРАЗЫ
    positive_phrases = [
        'на одном дыхании', 'актерская игра на высоте', 'лучший фильм года',
        'пересматривал несколько раз', 'остался под впечатлением', 
        'цепляет с первых минут', 'не отпускает до конца', 
        'операторская работа выше всяких похвал', 'берет за душу',
        'не оставляет равнодушным', 'глубокий смысл', 'философский подтекст',
        'потрясающая игра актеров', 'блестящий актерский состав', 
        'талантливая режиссура', 'качественный сценарий', 
        'продуманный до мелочей', 'внимание к деталям', 'смотрел на одном дыхании',
        'актеры играют превосходно', 'сюжет захватывающий', 'не оторваться'
    ]
    
    negative_phrases = [
        'зря потратил время', 'сюжетные дыры', 'персонажи картонные',
        'диалоги неестественные', 'спецэффекты выглядят дешево', 
        'концовка испортила весь фильм', 'первая половина интересная вторая разочаровала',
        'сюжет нелогичен', 'персонажи глупые', 'актеры не подходят для ролей',
        'режиссер не справился', 'можно было сократить', 'затянули',
        'скучно до зевоты', 'хотелось выключить', 'не оправдал ожиданий',
        'разочаровал полностью', 'ожидал гораздо большего',
        'бессмысленная трата времени', 'жаль потраченных денег', 'лучше бы поспал'
    ]
    
    # ПОДСЧЕТ ОЧКОВ
    positive_score = 0
    negative_score = 0
    
    # СЛОВА
    for word in positive_words:
        if word in text_lower:
            positive_score += 2
    
    for word in negative_words:
        if word in text_lower:
            negative_score += 2
    
    # ФРАЗЫ (высокий вес)
    for phrase in positive_phrases:
        if phrase in text_lower:
            positive_score += 3
    
    for phrase in negative_phrases:
        if phrase in text_lower:
            negative_score += 3
    
    # ЭМОЦИОНАЛЬНЫЕ СЛОВА
    emotional_positive = ['смеялся', 'плакал', 'восхищался', 'вдохновился', 'тронул']
    emotional_negative = ['скучал', 'злился', 'разочаровался', 'устал', 'заскучал']
    
    for word in emotional_positive:
        if word in text_lower:
            positive_score += 2
    
    for word in emotional_negative:
        if word in text_lower:
            negative_score += 2
    
    # УСИЛИТЕЛИ
    words = text_lower.split()
    for i, word in enumerate(words):
        if word in ['очень', 'крайне', 'невероятно', 'абсолютно', 'совершенно', 'просто']:
            if i + 1 < len(words):
                next_word = words[i + 1]
                if any(pos in next_word for pos in positive_words + strong_positive):
                    positive_score += 2
                elif any(neg in next_word for neg in negative_words + strong_negative):
                    negative_score += 2
    
    # ЛОГИКА ПРИНЯТИЯ РЕШЕНИЯ
    total_score = positive_score - negative_score
    
    # 1. ЛЮБОЙ позитивный сигнал → ПОЗИТИВНЫЙ
    if positive_score > 0 and negative_score == 0:
        confidence = min(0.7 + (positive_score * 0.08), 0.92)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    
    # 2. ЛЮБОЙ негативный сигнал → НЕГАТИВНЫЙ
    if negative_score > 0 and positive_score == 0:
        confidence = min(0.7 + (negative_score * 0.08), 0.92)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # 3. СМЕШАННЫЕ СИГНАЛЫ
    if positive_score > negative_score:
        confidence = 0.6 + ((positive_score - negative_score) * 0.1)
        return "ПОЗИТИВНЫЙ", min(confidence, 0.85), "🙂"
    
    if negative_score > positive_score:
        confidence = 0.6 + ((negative_score - positive_score) * 0.1)
        return "НЕГАТИВНЫЙ", min(confidence, 0.85), "😐"
    
    # 4. РАВНЫЕ ОЧКИ - СКЛОНЯЕМСЯ К НЕГАТИВУ (статистика)
    if positive_score == negative_score and positive_score > 0:
        return "НЕГАТИВНЫЙ", 0.6, "😐"
    
    # 5. ТОЛЬКО при явных нейтральных словах
    neutral_words = ['средне', 'нормально', 'обычно', 'стандартно', 'ничего особенного']
    if any(word in text_lower for word in neutral_words):
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # 6. ПО УМОЛЧАНИЮ - НЕГАТИВ (чаще встречается в отзывах)
    return "НЕГАТИВНЫЙ", 0.55, "😐"

# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def analyze_russian_review(russian_text):
    """Анализ одного отзыва"""
    try:
        sentiment, confidence, emotion = smart_sentiment_analyzer(russian_text)
        
        print(f"🔍 Анализ: '{russian_text[:60]}...' -> {sentiment} ({confidence:.2f})")
        
        return {
            'original_text': russian_text,
            'translated_text': 'Анализ выполнен умной системой',
            'sentiment_ru': sentiment,
            'confidence': float(confidence),
            'emotion': emotion
        }
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return {
            'original_text': russian_text,
            'translated_text': f'Ошибка: {str(e)}',
            'sentiment_ru': 'ОШИБКА',
            'confidence': 0.0,
            'emotion': '❌'
        }

def analyze_batch_reviews(texts):
    """Анализ пакета отзывов"""
    results = []
    statistics = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'errors': 0,
        'avg_confidence': 0
    }
    
    for text in texts:
        if text.strip():
            result = analyze_russian_review(text.strip())
            results.append(result)
            
            statistics['total'] += 1
            if result['sentiment_ru'] == 'ПОЗИТИВНЫЙ':
                statistics['positive'] += 1
            elif result['sentiment_ru'] == 'НЕГАТИВНЫЙ':
                statistics['negative'] += 1
            elif result['sentiment_ru'] == 'НЕОПРЕДЕЛЕНО':
                statistics['neutral'] += 1
            else:
                statistics['errors'] += 1
    
    if results:
        valid_confidences = [r['confidence'] for r in results if r['sentiment_ru'] != 'ОШИБКА']
        if valid_confidences:
            statistics['avg_confidence'] = sum(valid_confidences) / len(valid_confidences)
    
    return results, statistics

def create_text_statistics(statistics):
    """Создает текстовое представление статистики"""
    total = statistics['total']
    if total == 0:
        return "Нет данных для отображения"
    
    positive_percent = (statistics['positive'] / total) * 100 if total > 0 else 0
    negative_percent = (statistics['negative'] / total) * 100 if total > 0 else 0
    neutral_percent = (statistics['neutral'] / total) * 100 if total > 0 else 0
    
    text_stats = f"""
📊 СТАТИСТИКА АНАЛИЗА:

✅ Позитивные: {statistics['positive']} ({positive_percent:.1f}%)
❌ Негативные: {statistics['negative']} ({negative_percent:.1f}%) 
😐 Нейтральные: {statistics['neutral']} ({neutral_percent:.1f}%)
⚠️ Ошибки: {statistics['errors']}

📈 Всего отзывов: {total}
🎯 Средняя уверенность: {statistics['avg_confidence']:.1%}
"""
    
    return text_stats

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review_text = data['review'].strip()
        
        if not review_text:
            return jsonify({'success': False, 'error': 'Review is empty'}), 400
        
        result = analyze_russian_review(review_text)
        
        return jsonify({
            'success': True,
            'original_text': result['original_text'],
            'translated_text': result['translated_text'],
            'sentiment_ru': result['sentiment_ru'],
            'confidence': result['confidence'],
            'emotion': result['emotion']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.txt'):
            return jsonify({'success': False, 'error': 'Only .txt files allowed'}), 400
        
        # Читаем файл
        content = file.read().decode('utf-8')
        texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        print(f"📊 Анализируем {len(texts)} отзывов...")
        
        if len(texts) == 0:
            return jsonify({'success': False, 'error': 'File is empty'}), 400
        
        # Анализируем
        results, statistics = analyze_batch_reviews(texts)
        text_stats = create_text_statistics(statistics)
        
        print(f"✅ Анализ завершен. Обработано: {len(results)} отзывов")
        
        return jsonify({
            'success': True,
            'results': results,
            'statistics': statistics,
            'text_stats': text_stats,
            'processed_count': len(results)
        })
            
    except Exception as e:
        print(f"❌ Ошибка в analyze_batch: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/test')
def test():
    return jsonify({
        'message': 'Сервер работает с умной системой анализа!', 
        'status': 'OK',
        'analyzer_type': 'Smart Rule-Based',
        'version': '5.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
