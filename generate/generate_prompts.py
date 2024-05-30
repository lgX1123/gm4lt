from openai import OpenAI
import torchvision
import tqdm


def generate_prompts(dataset, data_num, labels):
    created_prompt_file_path = './data/Created_prompts/' + dataset + '.txt'
    
    client = OpenAI()
    for label in tqdm.tqdm(labels):
        for _ in range(data_num):
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output one prompt(30-50 words) \
                  to describe a picture of the object given by user about what it looks like."},
                {"role": "user", "content": f"The object: {label}."}
            ]
            )

            prompt = completion.choices[0].message.content
            with open(created_prompt_file_path, 'a') as file:
                file.write(f'{label}. ' + prompt + '\n')
        print(label + ' done!')
    

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=False)
    generate_prompts('cifar100', 500, dataset.classes)
