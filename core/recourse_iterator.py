import pandas as pd


class RecourseIterator:
    def __init__(self, recourse_method, preprocessor, certainty_cutoff=None, model=None):
        self.recourse_method = recourse_method
        self.certainty_cutoff = certainty_cutoff
        self.model = model
        self.preprocessor = preprocessor

    def iterate_k_recourse_paths(self, poi, max_iterations):
        all_instructions = self.recourse_method.get_all_recourse_instructions(poi)
        cfes = []
        for i in range(len(all_instructions)):
            instructions = all_instructions[all_instructions.index==i]
            cfe = self.preprocessor.interpret_instructions(poi, instructions)
            cfes.append(cfe)
        paths = []
        for dir_index, cfe in enumerate(cfes):
            path = [poi] + self.iterate_recourse_path(cfe, dir_index, max_iterations - 1)
            paths.append(pd.concat(path).reset_index(drop=True))
        return paths

    def iterate_recourse_path(self, poi, dir_index, max_iterations):
        path = [poi]
        for i in range(max_iterations):
            if self.certainty_cutoff and self.check_model_certainty(self.preprocessor.transform(poi)) > self.certainty_cutoff:
                break
            instructions = self.recourse_method.get_kth_recourse_instructions(poi, dir_index)
            poi = self.preprocessor.interpret_instructions(poi, instructions)
            path.append(poi)
        return path

    def check_model_certainty(self, transformed_poi):
        proba = self.model.predict_proba(transformed_poi)[0]
        return proba

