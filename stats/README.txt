
class-cond.pkl:
	dumped 3 times in the following order
	[class, legibility, language] - conditioned coexistence data.
	“class”                  :   str     # ‘machine printed’ or ‘handwritten’ or ‘others’
	“legibility”             :   str     # ‘legible’ or ‘illegible’
	“language”               :   str     # ‘english’ or ‘not english’ or ‘na’

	We calculate coexistence rate for each annotation
	as (# of words in annotation that appears in caption)/(# of words in annotations, no duplicates).

