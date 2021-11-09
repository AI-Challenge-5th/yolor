import json

def evaluate():
    test_dict = [
            {"framework": "pytorch"},
            
            {"parameters":2411414},
            
            {
                "image_id":9070,
                "category_id":3,
                "bbox":[60.678, 63.976, 98.278, 38.435],
                "score":0.92895
            }
            ]
    with open('answersheet_4_04_nuxlear.json','w', encoding='utf-8') as f:
        json.dump(test_dict, f, ensure_ascii=False, indent=4)
        print("json creation success!!")
    
    
if __name__ == "__main__":     
    evaluate()
