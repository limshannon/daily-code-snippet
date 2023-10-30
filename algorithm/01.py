vowels = ['a', 'e', 'i', 'o', 'u', 'y'] 

def check_for_pattern(pattern, source, start_index):
   for offset in range(len(pattern)):
       if pattern[offset] == '0':
           if source[start_index + offset] not in vowels:
               return 0
       else:
           if source[start_index + offset] in vowels:
               return 0
   return 1
  def solution(pattern, source):
   answer = 0
   for start_index in range(len(source) - len(pattern) + 1):
       answer += check_for_pattern(pattern, source, start_index)
   return answer
