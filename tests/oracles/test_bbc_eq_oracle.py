from typing import Any, Tuple
import unittest

from aalpy.automata import Dfa, DfaState, MealyState, MealyMachine
from aalpy.base import Oracle
from aalpy.base.Automaton import Automaton, InputType
from aalpy.learning_algs import run_Lstar
from aalpy.model_checking_oracles import IUOBugDfaModelCheckingOracle
from aalpy.oracles import BBCEqOracle, WMethodEqOracle
from aalpy.SULs import AutomatonSUL
from tests.oracles.test_baseOracle import BaseOracleTests


def get_accepting_oracle() -> Oracle:
    """
    Create an Oracle which never returns a counterexample
    """
    class AcceptingOracle(Oracle):
        def __init__(self):
            super().__init__(None, None)

        def find_cex(self, hypothesis: Automaton) -> Tuple[InputType, ...] | None:
            return None

    return AcceptingOracle()


def precise_word_mealy(word: Tuple[str], alphabet=('a', 'b')):
    """
    Mealy machine that outputs len(word) - 1 `o`s followed by an `x` and then only 'o's for the word
    and only `o`s for any other input sequence.
    """
    length = len(word)
    sink = MealyState("sink")
    sink.transitions = {a: sink for a in alphabet}
    sink.output_fun = {a: 'o' for a in alphabet}
    states = [MealyState(f's_{word[:i]}') for i in range(length)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].output_fun = {a: 'o' for a in alphabet}
        if i < length - 1:
            states[i].transitions[word[i]] = states[i + 1]
        else:
            states[i].output_fun[word[i]] = 'x'
    return MealyMachine(states[0], states)

def precise_word_dfa(word: Tuple[str], alphabet=('a', 'b')):
    """
    Dfa which accepts the input sequences that are prefixed by the given word.
    """
    length = len(word)
    sink = DfaState("sink", False)
    sink.transitions = {a: sink for a in alphabet}
    states = [DfaState(f's_{word[:i]}', i == length) for i in range(length + 1)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].transitions[word[i]] = states[i + 1]
    states[length].transitions = {a: sink for a in alphabet}
    return Dfa(states[0], states)


def dfa_input_from_mealy_input(input: Any) -> Tuple[str,Any]:
    return ("I", input)

def dfa_output_from_mealy_output(output: Any) -> Tuple[str,Any]:
    return ("O", output)

def is_dfa_input(letter: Tuple[str,Any]) -> bool:
    return letter[0] == "I"

def mealy_letter_from_dfa_letter(letter: Tuple[str,Any]) -> Any:
    return letter[1]


class BBCEqOracleTests(BaseOracleTests):

    def test_values_are_bound_properly(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        sul = AutomatonSUL(mealy)

        base_oracle = get_accepting_oracle()
        oracle = BBCEqOracle(
            alphabet=mealy.get_input_alphabet(),
            sul=sul,
            property_oracles=dict(),
            eq_oracle=base_oracle
        )

        self.assertEqual(oracle.alphabet, base_oracle.alphabet)
        self.assertEqual(oracle.sul, base_oracle.sul)
        self.assertEqual(oracle.num_queries, base_oracle.num_queries)
        self.assertEqual(oracle.num_steps, base_oracle.num_steps)

        oracle.alphabet = ["h", "e", "l", "l", "o"]
        oracle.sul = AutomatonSUL(mealy)
        oracle.num_queries = 12345
        oracle.num_steps = 54321
        self.assertEqual(base_oracle.alphabet, oracle.alphabet)
        self.assertEqual(base_oracle.sul, oracle.sul)
        self.assertEqual(base_oracle.num_queries, oracle.num_queries)
        self.assertEqual(base_oracle.num_steps, oracle.num_steps)

    def test_valid_oracle_when_no_properties(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        base_oracle = WMethodEqOracle(alphabet, learning_sul, len(learning_sul.automaton.states) + 1)
        oracle = BBCEqOracle(
            alphabet=alphabet,
            sul=learning_sul,
            property_oracles=dict(),
            eq_oracle=base_oracle
        )

        self.validate_eq_oracle(alphabet, oracle, learning_sul, validation_sul)

    def test_no_prop_cex_found_if_none_exist(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        sul = AutomatonSUL(mealy)

        base_oracle = WMethodEqOracle(mealy.get_input_alphabet(), sul, len(mealy.states) + 1)
        def violation_callback(label: str, cex: tuple) -> None:
            # The property has no counterexample that could have been confirmed against the SUT, so the callback may not have been called with a counterexample
            assert False

        oracle = BBCEqOracle(
            alphabet=mealy.get_input_alphabet(),
            sul=sul,
            property_oracles={
                "SatisfiedProperty": get_accepting_oracle()
            },
            eq_oracle=base_oracle,
            property_violation_callback=violation_callback
        )

        hyp_mealy = MealyMachine.from_state_setup(mealy.to_state_setup())
        self.assertIsNone(oracle.find_cex(hyp_mealy))

    def test_prop_cex_found_if_and_only_if_it_exist(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        sul = AutomatonSUL(mealy)

        violation_cex = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex

            assert label == "ViolatedProperty"
            violation_cex = cex
        base_oracle = WMethodEqOracle(mealy.get_input_alphabet(), sul, len(mealy.states) + 1)

        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )
        violated_prop_mc_oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        oracle = BBCEqOracle(
            alphabet=mealy.get_input_alphabet(),
            sul=sul,
            property_oracles={
                "SatisfiedProperty": get_accepting_oracle(),
                "ViolatedProperty": violated_prop_mc_oracle
            },
            eq_oracle=base_oracle,
            property_violation_callback=violation_callback
        )

        hyp_mealy = MealyMachine.from_state_setup(mealy.to_state_setup())
        self.assertIsNone(oracle.find_cex(hyp_mealy))
        self.assertEqual(violation_cex, violated_prop_mc_oracle.find_cex(hyp_mealy))

        # The violation_cex passed to the callback should yield the same outputs in the SUL as in the Mealy machine
        self.assertEqual(sul.query(violation_cex), mealy.execute_sequence(mealy.initial_state, violation_cex))

    def test_find_prop_cex_to_be_used_for_refinement(self):
        sul_mealy = MealyMachine.from_state_setup({
            "q0": {"i": ("o0", "q1")},
            "q1": {"i": ("o1", "q1")}
        })
        initial_hyp_mealy = MealyMachine.from_state_setup({
            "h0": {"i": ("o1", "h0")}
        })
        prop_bug_dfa = Dfa.from_state_setup({
            "p": (False, {
                dfa_input_from_mealy_input("i"): "p",
                dfa_output_from_mealy_output("o0"): "p",
                dfa_output_from_mealy_output("o1"): "b"
            }),
            "b": (True, {
                dfa_input_from_mealy_input("i"): "p",
                dfa_output_from_mealy_output("o0"): "p",
                dfa_output_from_mealy_output("o1"): "p"}),
        })

        prop_mc_oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        def violation_callback(label: str, cex: tuple) -> None:
            # The property has no counterexample that could have been confirmed against the SUT, so the callback may not have been called with a counterexample
            assert False

        sul = AutomatonSUL(sul_mealy)
        oracle = BBCEqOracle(
            alphabet=sul_mealy.get_input_alphabet(),
            sul=sul,
            property_oracles={
                "Property": prop_mc_oracle
            },
            eq_oracle=get_accepting_oracle(),
            property_violation_callback=violation_callback
        )

        cex = oracle.find_cex(initial_hyp_mealy)
        # cex should be None, since initial_hyp_mealy satisfies the property
        self.assertIsNotNone(cex)
        # The SUL and the hypothesis should have distinct outputs for cex, since cex should be a valid counterexample for hypothesis refinement
        self.assertNotEqual(sul.query(cex), initial_hyp_mealy.execute_sequence(initial_hyp_mealy.initial_state, cex))

    def test_prop_cex_is_not_found_before_this_is_possible_for_the_hyp(self):
        sul_mealy = MealyMachine.from_state_setup({
            "q0": {"i": ("o0", "q1")},
            "q1": {"i": ("o1", "q1")}
        })
        initial_hyp_mealy = MealyMachine.from_state_setup({
            "h0": {"i": ("o0", "h0")}
        })
        prop_bug_dfa = Dfa.from_state_setup({
            "p": (False, {
                dfa_input_from_mealy_input("i"): "p",
                dfa_output_from_mealy_output("o0"): "p",
                dfa_output_from_mealy_output("o1"): "b"
            }),
            "b": (True, {
                dfa_input_from_mealy_input("i"): "p",
                dfa_output_from_mealy_output("o0"): "p",
                dfa_output_from_mealy_output("o1"): "p"}),
        })

        prop_mc_oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        def violation_callback(label: str, cex: tuple) -> None:
            # The property has no counterexample that could have been confirmed against the SUT, so the callback may not have been called with a counterexample
            assert False

        sul = AutomatonSUL(sul_mealy)
        oracle = BBCEqOracle(
            alphabet=sul_mealy.get_input_alphabet(),
            sul=sul,
            property_oracles={
                "Property": prop_mc_oracle
            },
            eq_oracle=get_accepting_oracle(),
            property_violation_callback=violation_callback
        )

        cex = oracle.find_cex(initial_hyp_mealy)
        # Cex should be None, since initial_hyp_mealy satisfies the property
        self.assertIsNone(cex)

    def test_valid_oracle_and_correct_values_when_satisfied_and_violated_properties(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )

        learning_sul = AutomatonSUL(mealy)
        validation_sul = AutomatonSUL(mealy)

        violation_cex = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex

            assert label == "ViolatedProperty"
            violation_cex = cex
        base_oracle = WMethodEqOracle(mealy.get_input_alphabet(), learning_sul, len(mealy.states) + 1)

        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )
        violated_prop_mc_oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        oracle = BBCEqOracle(
            alphabet=mealy.get_input_alphabet(),
            sul=learning_sul,
            property_oracles={
                "SatisfiedProperty": get_accepting_oracle(),
                "ViolatedProperty": violated_prop_mc_oracle
            },
            eq_oracle=base_oracle,
            property_violation_callback=violation_callback
        )
        self.assertEqual(oracle.alphabet, base_oracle.alphabet)
        self.assertEqual(oracle.sul, base_oracle.sul)
        self.assertEqual(oracle.num_queries, base_oracle.num_queries)
        self.assertEqual(oracle.num_steps, base_oracle.num_steps)

        learned_model = run_Lstar(
            mealy.get_input_alphabet(), learning_sul, oracle, 'mealy', print_level=2)
        validation_eq_oracle = WMethodEqOracle(
            mealy.get_input_alphabet(), validation_sul, max_number_of_states=len(learned_model.states) + 2)
        self.assertIsNone(validation_eq_oracle.find_cex(
            learned_model), "Counterexample found by WMethodEqOracle")

        # The violation_cex passed to the callback should yield the same outputs in the SUL as in the Mealy machine
        self.assertIsNotNone(violation_cex)
        self.assertEqual(learning_sul.query(violation_cex), mealy.execute_sequence(mealy.initial_state, violation_cex))
        self.assertEqual(violation_cex, violated_prop_mc_oracle.find_cex(mealy))

        self.assertEqual(oracle.alphabet, base_oracle.alphabet)
        self.assertEqual(oracle.sul, base_oracle.sul)
        self.assertEqual(oracle.num_queries, base_oracle.num_queries)
        self.assertEqual(oracle.num_steps, base_oracle.num_steps)

    def test_violation_callback_is_optional(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )

        learning_sul = AutomatonSUL(mealy)
        validation_sul = AutomatonSUL(mealy)

        base_oracle = WMethodEqOracle(mealy.get_input_alphabet(), learning_sul, len(mealy.states) + 1)

        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )
        violated_prop_mc_oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        oracle = BBCEqOracle(
            alphabet=mealy.get_input_alphabet(),
            sul=learning_sul,
            property_oracles={
                "ViolatedProperty": violated_prop_mc_oracle
            },
            eq_oracle=base_oracle
        )

        hyp_mealy = MealyMachine.from_state_setup(mealy.to_state_setup())
        oracle.find_cex(hyp_mealy)

    def test_properties_with_confirmed_sut_counterexamples_are_excluded_until_reenabled(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        sul = AutomatonSUL(mealy)

        violation_cex = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex

            assert label == "ViolatedProperty"
            violation_cex = cex
        base_oracle = WMethodEqOracle(mealy.get_input_alphabet(), sul, len(mealy.states) + 1)

        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )
        violated_prop_mc_oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        oracle = BBCEqOracle(
            alphabet=mealy.get_input_alphabet(),
            sul=sul,
            property_oracles={
                "SatisfiedProperty": get_accepting_oracle(),
                "ViolatedProperty": violated_prop_mc_oracle
            },
            eq_oracle=base_oracle,
            property_violation_callback=violation_callback
        )

        hyp_mealy = MealyMachine.from_state_setup(mealy.to_state_setup())
        oracle.find_cex(hyp_mealy)
        self.assertIsNotNone(violation_cex)
        violation_cex = None

        oracle.find_cex(hyp_mealy)
        self.assertIsNone(violation_cex)

        oracle.forget_all_property_violations()
        oracle.find_cex(hyp_mealy)
        self.assertIsNotNone(violation_cex)


if __name__ == '__main__':
    unittest.main()
