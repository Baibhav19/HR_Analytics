import os

def template():
    # creating basic project structure
    
    dirs = [
        os.path.join('data', 'raw'),
        os.path.join('data', 'processed'),
        'src',
        'saved_models',
        'notebooks',
        'report'
    ]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, '.gitkeep') ,'w') as f:
            pass
    
    
if __name__ == '__main__':
	template()