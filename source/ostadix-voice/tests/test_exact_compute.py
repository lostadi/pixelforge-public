import unittest
from exact_compute import calculate

class ExactTests(unittest.TestCase):
    def test_exact_results(self):
        for expression, expected in [('0.1+0.2','3/10'), ('(18+6)/8','3'),
                                     ('2**-3','1/8'), ('-7//3','-3'), ('-7%3','2')]:
            with self.subTest(expression=expression):
                result=calculate(expression)
                self.assertEqual(result['exact'],expected)
                self.assertEqual(result['model_calls'],0)
    def test_rejects_code_and_unbounded_inputs(self):
        for expression in ['__import__("os")', 'True', 'x+1', '2**33', '2**0.5',
                           '1e300', '1/0', '(99999999999999999999**32)**32',
                           '+'.join(['1']*40), '1'*201]:
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError,SyntaxError,ZeroDivisionError)):
                    calculate(expression)
