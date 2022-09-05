from nusantara import NusantaraMetadata, NusantaraConfigHelper
from nusantara.utils.constants import Tasks

if __name__ == "__main__":

    conhelps = NusantaraConfigHelper()
    print('All Configs')
    print(conhelps)

    # filter and load datasets
    # ====================================================================
    print('Retrieve SMSA')
    print([helper for helper in conhelps.filtered(lambda x: ("smsa" in x.dataset_name and x.is_nusantara_schema))])
    smsa_datasets = [
        helper.load_dataset()
        for helper in conhelps.filtered(
            lambda x: ("smsa" in x.dataset_name and x.is_nusantara_schema)
        )
    ]
    print(smsa_datasets)

    # examples of other filters
    # ====================================================================

    # get all source schema config helpers
    print('Source datasets')
    source_helpers = conhelps.filtered(lambda x: x.config.schema == "source")
    print(source_helpers)
    
    # get all nusantara config helpers
    print('Nusantara datasets')
    nusantara_helpers = conhelps.filtered(lambda x: x.is_nusantara_schema)
    print(nusantara_helpers)

    # nusantara NER public tasks
    print('Nusantara NER public datasets')
    nc_ner_public_helpers = conhelps.filtered(
        lambda x: (
            x.is_nusantara_schema
            and Tasks.NAMED_ENTITY_RECOGNITION in x.tasks
            and not x.is_local
        )
    )
    print(nc_ner_public_helpers)

    # indolem datasets
    print('IndoLEM datasets')
    nc_indolem_helpers = conhelps.filtered(
        lambda x: ("indolem" in x.dataset_name and x.is_nusantara_schema)
    )
    print(nc_indolem_helpers)